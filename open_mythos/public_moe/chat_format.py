"""多轮消息格式化工具。"""
from __future__ import annotations

from typing import Any, Iterable, Mapping


def normalize_messages(messages: Any) -> list[dict[str, str]]:
    """把外部传入的 messages 规整成 OpenAI 风格列表。"""
    if not isinstance(messages, list):
        return []
    out: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role", "user")).strip().lower() or "user"
        if role not in {"system", "user", "assistant", "tool"}:
            role = "user"
        out.append({"role": role, "content": str(item.get("content", ""))})
    return out


def format_messages(messages: Iterable[Mapping[str, str]], add_assistant_prefix: bool = True) -> str:
    """把 OpenAI 风格 messages 转成稳定的纯文本提示词。

    该格式不依赖具体 tokenizer，适合 tiny true-MoE 与公开 Hugging Face MoE 共用。
    """
    parts: list[str] = []
    for msg in normalize_messages(list(messages)):
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<{role}>\n{content}\n</{role}>")
    if add_assistant_prefix:
        parts.append("<assistant>\n")
    return "\n".join(parts)
