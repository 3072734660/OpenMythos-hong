#!/usr/bin/env python
"""不依赖 Ollama 的本地公开 MoE HTTP 服务。"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from open_mythos.public_moe.runtime import PublicMoERuntime
from open_mythos.public_moe.chat_format import normalize_messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/family_small_moe_3050ti_4g.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()

    runtime = PublicMoERuntime(args.config)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, payload: dict, status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(runtime.health())
            else:
                self._send({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            try:
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
                max_new_tokens = int(body.get("max_tokens", body.get("max_new_tokens", 64)))
                temperature = float(body.get("temperature", 0.7))
                top_k = int(body.get("top_k", 50))

                if self.path in {"/generate", "/v1/completions"}:
                    result = runtime.generate(str(body.get("prompt", "")), max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
                elif self.path == "/v1/chat/completions":
                    result = runtime.generate_from_messages(normalize_messages(body.get("messages", [])), max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
                else:
                    self._send({"error": "not found"}, 404)
                    return

                meta = {
                    "cache_hit": result.cache_hit,
                    "prefix_cache_hit": result.prefix_cache_hit,
                    "prefix_cache_tokens": result.prefix_cache_tokens,
                    "prefill_tokens": result.prefill_tokens,
                    "generated_tokens": result.generated_tokens,
                }
                if self.path == "/v1/chat/completions":
                    self._send({"choices": [{"message": {"role": "assistant", "content": result.text}, "finish_reason": "stop"}], **meta})
                elif self.path == "/v1/completions":
                    self._send({"choices": [{"text": result.text, "finish_reason": "stop"}], **meta})
                else:
                    self._send({"text": result.text, **meta, "elapsed_sec": result.elapsed_sec})
            except Exception as exc:
                self._send({"error": str(exc)}, 500)

        def log_message(self, fmt: str, *args) -> None:
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"status": "serving", "host": args.host, "port": args.port, "health": runtime.health()}, ensure_ascii=False, indent=2), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
