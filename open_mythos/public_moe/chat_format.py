"""多轮消息格式化工具。"""
from __future__ import annotations

from typing import Iterable, Mapping


def format_messages(messages: Iterable[Mapping[str, str]], add_assistant_prefix: bool = True) -> str:
    """把 OpenAI 风格 messages 转成稳定的纯文本提示词。

    该格式不依赖具体 tokenizer，适合 tiny true-MoE 与公开 Hugging Face MoE 共用。
    """
    parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "user")).strip().lower() or "user"
        content = str(msg.get("content", ""))
        if role not in {"system", "user", "assistant", "tool"}:
            role = "user"
        parts.append(f"<{role}>\n{content}\n</{role}>")
    if add_assistant_prefix:
        parts.append("<assistant>\n")
    return "\n".join(parts)
