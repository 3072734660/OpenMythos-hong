"""tiny true-MoE causal LM，用于调试三模型统一流程。"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class TinyByteTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str) -> list[int]:
        return [b + 2 for b in text.encode("utf-8", errors="replace")] + [self.eos_token_id]

    def decode(self, ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        bs = bytearray()
        for i in ids:
            if i in {self.pad_token_id, self.eos_token_id} and skip_special_tokens:
                continue
            if 2 <= int(i) <= 257:
                bs.append(int(i) - 2)
        return bs.decode("utf-8", errors="replace")

    def __call__(self, text: str, return_tensors: str | None = None) -> dict[str, torch.Tensor] | list[int]:
        ids = self.encode(text)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return ids


@dataclass
class TinyTrueMoEConfig:
    vocab_size: int = 258
    dim: int = 32
    n_layers: int = 2
    n_heads: int = 4
    n_experts: int = 4
    top_k: int = 2
    expert_hidden_dim: int = 64
    max_seq_len: int = 896


class TinyMoE(nn.Module):
    def __init__(self, dim: int, n_experts: int, top_k: int, hidden_dim: int):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router(x)
        weights, idx = torch.topk(torch.softmax(logits, dim=-1), k=self.top_k, dim=-1)
        out = torch.zeros_like(x)
        for slot in range(self.top_k):
            expert_ids = idx[..., slot]
            w = weights[..., slot].unsqueeze(-1)
            for expert_id, expert in enumerate(self.experts):
                mask = expert_ids == expert_id
                if mask.any():
                    out[mask] += expert(x[mask]) * w[mask]
        return out


class TinyBlock(nn.Module):
    def __init__(self, cfg: TinyTrueMoEConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.dim)
        self.attn = nn.MultiheadAttention(cfg.dim, cfg.n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(cfg.dim)
        self.moe = TinyMoE(cfg.dim, cfg.n_experts, cfg.top_k, cfg.expert_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.moe(self.norm2(x))
        return x


class TinyTrueMoEForCausalLM(nn.Module):
    def __init__(self, cfg: TinyTrueMoEConfig):
        super().__init__()
        self.config = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.dim)
        self.blocks = nn.ModuleList([TinyBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, **_: object):
        b, t = input_ids.shape
        if t > self.config.max_seq_len:
            input_ids = input_ids[:, -self.config.max_seq_len:]
            t = input_ids.shape[1]
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
        x = self.embed(input_ids) + self.pos(pos)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)), labels[:, 1:].contiguous().view(-1), ignore_index=-100)
        return type("TinyOutput", (), {"logits": logits, "loss": loss})()

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64, do_sample: bool = False, temperature: float = 1.0, top_k: int = 0, **_: object) -> torch.Tensor:
        out = input_ids
        for _step in range(max_new_tokens):
            logits = self(out).logits[:, -1, :]
            if do_sample and temperature > 0:
                logits = logits / temperature
                if top_k and top_k > 0:
                    v, ix = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
                    probs = torch.softmax(v, dim=-1)
                    next_id = ix.gather(-1, torch.multinomial(probs, 1))
                else:
                    next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            out = torch.cat([out, next_id], dim=1)
            if bool((next_id == 1).all()):
                break
        return out
