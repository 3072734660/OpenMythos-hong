"""公开预训练 MoE 与 tiny true-MoE 的统一本地 runtime。"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from .chat_format import format_messages


@dataclass
class GenerationResult:
    text: str
    cache_hit: bool = False
    prefix_cache_hit: bool = False
    prefix_cache_tokens: int = 0
    prefill_tokens: int = 0
    generated_tokens: int = 0
    elapsed_sec: float = 0.0


class PublicMoERuntime:
    """统一加载 tiny true-MoE 或 Hugging Face 公开 MoE。

    初始化仓库中的实现重视清晰与可维护性。大型模型的高性能推理可后续替换为 vLLM/SGLang。
    """

    def __init__(self, config_path: str | Path):
        self.config_path = str(config_path)
        self.config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self.cfg = self.config.get("public_pretrained_moe", {})
        self.response_cache_size = int(self.cfg.get("response_cache_size", 0) or 0)
        self.response_cache: dict[str, GenerationResult] = {}
        self.response_cache_hits = 0
        self.response_cache_misses = 0
        self.prefix_cache_enabled = bool(self.cfg.get("enable_prefix_cache", True))
        self.prefix_cache: dict[str, dict[str, Any]] = {}
        self.prefix_cache_hits = 0
        self.prefix_cache_misses = 0
        self.prefix_cache_reused_tokens = 0
        self.prefix_cache_prefill_tokens = 0
        self.backend = str(self.cfg.get("backend", "hf_transformers"))
        self.device = str(self.cfg.get("device", "cpu"))
        self.quantize = str(self.cfg.get("quantize", "none"))
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self) -> None:
        if self.backend == "tiny_true_moe":
            from .tiny_true_moe import TinyTrueMoEConfig, TinyTrueMoEForCausalLM, TinyByteTokenizer

            tcfg = self.config.get("training", {})
            model_cfg = TinyTrueMoEConfig(
                vocab_size=258,
                dim=int(tcfg.get("dim", 32)),
                n_layers=int(tcfg.get("n_layers", 2)),
                n_heads=int(tcfg.get("n_heads", 4)),
                n_experts=int(tcfg.get("n_experts", 4)),
                top_k=int(tcfg.get("top_k", 2)),
                expert_hidden_dim=int(tcfg.get("expert_hidden_dim", 64)),
                max_seq_len=int(tcfg.get("max_seq_len", 896)),
            )
            self.model = TinyTrueMoEForCausalLM(model_cfg)
            self.tokenizer = TinyByteTokenizer()
            if self.quantize == "int8_dynamic":
                self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
            self.model.eval()
            self.model.to("cpu")
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError("请先安装 transformers/accelerate/bitsandbytes 等依赖") from exc

        model_path = self.cfg.get("local_model_path") or self.cfg.get("model_id")
        local_files_only = bool(self.cfg.get("local_files_only", True))
        trust_remote_code = bool(self.cfg.get("trust_remote_code", False))
        quantization_config = None
        load_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
            "device_map": self.cfg.get("device_map", "auto"),
            "low_cpu_mem_usage": bool(self.cfg.get("low_cpu_mem_usage", True)),
        }
        if isinstance(self.cfg.get("max_memory"), dict):
            load_kwargs["max_memory"] = self.cfg["max_memory"]
        if self.quantize == "4bit_bnb":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.cfg.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=bool(self.cfg.get("bnb_4bit_use_double_quant", True)),
            )
        elif self.quantize == "8bit_bnb":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self.model.eval()

    def _cache_key(self, prompt: str, max_new_tokens: int, temperature: float, top_k: int) -> str:
        raw = json.dumps({"p": prompt, "m": max_new_tokens, "t": temperature, "k": top_k}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _remember_response(self, key: str, value: GenerationResult) -> None:
        if self.response_cache_size <= 0:
            return
        if len(self.response_cache) >= self.response_cache_size:
            self.response_cache.pop(next(iter(self.response_cache)))
        self.response_cache[key] = value

    def health(self) -> dict[str, Any]:
        n_params = 0
        try:
            n_params = sum(p.numel() for p in self.model.parameters())
        except Exception:
            pass
        return {
            "status": "ok",
            "backend": self.backend,
            "device": self.device,
            "quantize": self.quantize,
            "params": n_params,
            "response_cache_size": self.response_cache_size,
            "response_cache_hits": self.response_cache_hits,
            "response_cache_misses": self.response_cache_misses,
            "prefix_cache_enabled": self.prefix_cache_enabled,
            "prefix_cache_entries": len(self.prefix_cache),
            "prefix_cache_hits": self.prefix_cache_hits,
            "prefix_cache_misses": self.prefix_cache_misses,
            "prefix_cache_reused_tokens": self.prefix_cache_reused_tokens,
            "prefix_cache_prefill_tokens": self.prefix_cache_prefill_tokens,
        }

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.7, top_k: int = 50) -> GenerationResult:
        started = time.time()
        key = self._cache_key(prompt, max_new_tokens, temperature, top_k)
        if key in self.response_cache:
            self.response_cache_hits += 1
            cached = self.response_cache[key]
            return GenerationResult(**{**cached.__dict__, "cache_hit": True, "elapsed_sec": time.time() - started})
        self.response_cache_misses += 1

        encoded = self.tokenizer(prompt, return_tensors="pt") if callable(self.tokenizer) else self.tokenizer.encode(prompt)
        if isinstance(encoded, dict):
            input_ids = encoded["input_ids"]
        else:
            input_ids = torch.tensor([encoded], dtype=torch.long)
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0, "use_cache": True}
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        if top_k > 0:
            gen_kwargs["top_k"] = top_k
        out = self.model.generate(input_ids=input_ids, **gen_kwargs)
        new_ids = out[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True) if hasattr(self.tokenizer, "decode") else self.tokenizer.decode(new_ids.tolist())
        result = GenerationResult(text=text, prefill_tokens=int(input_ids.shape[1]), generated_tokens=int(new_ids.numel()), elapsed_sec=time.time() - started)
        self._remember_response(key, result)
        return result

    def generate_from_messages(self, messages: list[dict[str, str]], **kwargs: Any) -> GenerationResult:
        return self.generate(format_messages(messages), **kwargs)
