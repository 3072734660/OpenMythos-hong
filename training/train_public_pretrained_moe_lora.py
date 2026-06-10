#!/usr/bin/env python
"""公开预训练 MoE 的 LoRA/QLoRA 训练入口。

本脚本是初始化版训练入口，不包含任何内置数据生成或评测逻辑。
用户需要显式提供自己的 JSONL 训练数据。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from open_mythos.public_moe.chat_format import format_messages, normalize_messages


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"训练数据为空：{path}")
    return rows


def build_prompt_response(row: dict[str, Any]) -> tuple[str, str]:
    if "messages" in row:
        prompt = format_messages(normalize_messages(row["messages"]), add_assistant_prefix=True)
        response = str(row.get("response", row.get("answer", "")))
    else:
        prompt = str(row.get("prompt", row.get("text", "")))
        response = str(row.get("response", row.get("answer", "")))
    return prompt, response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/family_small_moe_3050ti_4g.yaml")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", default="checkpoints/public_moe_lora")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pcfg = cfg.get("public_pretrained_moe", {})
    tcfg = cfg.get("training", {})
    max_steps = int(args.max_steps or tcfg.get("max_steps", 1000))
    batch_size = int(args.micro_batch_size or tcfg.get("micro_batch_size", 1))
    max_length = int(tcfg.get("max_length", 256))
    lr = float(tcfg.get("learning_rate", 1e-4))

    rows = load_jsonl(args.train_jsonl)
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run",
            "config": args.config,
            "train_rows": len(rows),
            "model_id": pcfg.get("model_id"),
            "local_model_path": pcfg.get("local_model_path"),
            "max_steps": max_steps,
            "batch_size": batch_size,
            "max_length": max_length,
        }, ensure_ascii=False, indent=2))
        return

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except Exception as exc:
        raise RuntimeError("请安装 transformers peft accelerate bitsandbytes 后再训练") from exc

    model_path = os.path.expandvars(str(pcfg.get("local_model_path") or pcfg.get("model_id")))
    local_files_only = bool(pcfg.get("local_files_only", True))
    trust_remote_code = bool(pcfg.get("trust_remote_code", False))
    quantize = str(pcfg.get("quantize", "none"))
    load_kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
        "device_map": pcfg.get("device_map", "auto"),
        "low_cpu_mem_usage": bool(pcfg.get("low_cpu_mem_usage", True)),
    }
    if isinstance(pcfg.get("max_memory"), dict):
        load_kwargs["max_memory"] = pcfg["max_memory"]
    if quantize == "4bit_bnb":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=pcfg.get("bnb_4bit_quant_type", "nf4"), bnb_4bit_use_double_quant=bool(pcfg.get("bnb_4bit_use_double_quant", True)))
    elif quantize == "8bit_bnb":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    if quantize in {"4bit_bnb", "8bit_bnb"}:
        model = prepare_model_for_kbit_training(model)
    if bool(pcfg.get("gradient_checkpointing", False)) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=int(pcfg.get("lora_r", 4)),
        lora_alpha=int(pcfg.get("lora_alpha", 8)),
        lora_dropout=float(pcfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grad_accum = int(tcfg.get("gradient_accumulation_steps", 1))
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    while global_step < max_steps:
        total_loss = 0.0
        for _ in range(grad_accum):
            row = rows[global_step % len(rows)]
            prompt, response = build_prompt_response(row)
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full = tokenizer(prompt + response + (tokenizer.eos_token or ""), truncation=True, max_length=max_length, return_tensors="pt")
            input_ids = full["input_ids"].to(model.device)
            labels = input_ids.clone()
            cut = min(len(prompt_ids), labels.shape[1])
            labels[:, :cut] = -100
            loss = model(input_ids=input_ids, labels=labels).loss / grad_accum
            loss.backward()
            total_loss += float(loss.detach().cpu())
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.get("max_grad_norm", 1.0)))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        if global_step % int(tcfg.get("save_steps", 50)) == 0 or global_step == max_steps:
            save_dir = out_dir / f"step_{global_step:06d}"
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
        print(json.dumps({"step": global_step, "loss": total_loss}, ensure_ascii=False), flush=True)

    print(json.dumps({"status": "ok", "output_dir": str(out_dir), "steps": global_step}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
