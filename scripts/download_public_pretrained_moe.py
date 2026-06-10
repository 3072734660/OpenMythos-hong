#!/usr/bin/env python
"""下载公开预训练 MoE 模型到本地目录，供离线训练/服务使用。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="allenai/OLMoE-1B-7B-0125")
    parser.add_argument("--local-dir", default="models/OLMoE-1B-7B-0125")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = {
        "model_id": args.model_id,
        "local_dir": args.local_dir,
        "revision": args.revision,
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps({"status": "dry_run", **report}, ensure_ascii=False, indent=2))
        return

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("请先安装 huggingface_hub：pip install huggingface_hub") from exc

    resolved = snapshot_download(
        repo_id=args.model_id,
        local_dir=args.local_dir,
        revision=args.revision,
        local_dir_use_symlinks=False,
    )
    sample_files = [str(p.relative_to(resolved)) for p in Path(resolved).rglob("*") if p.is_file()][:50]
    print(json.dumps({"status": "ok", **report, "resolved_path": resolved, "sample_files": sample_files}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
