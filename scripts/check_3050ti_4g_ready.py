#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/family_small_moe_3050ti_4g.yaml')
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--require-weights', action='store_true')
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    pcfg = cfg.get('public_pretrained_moe', {})
    errors = []

    cuda_available = False
    device_count = 0
    device_name = None
    memory_bytes = 0
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            device_count = int(torch.cuda.device_count())
            device_name = torch.cuda.get_device_name(0)
            memory_bytes = int(torch.cuda.get_device_properties(0).total_memory)
    except Exception as exc:
        errors.append('torch check failed: ' + str(exc))

    if args.require_cuda and not cuda_available:
        errors.append('cuda is not available')

    model_path = str(pcfg.get('local_model_path') or '')
    model_exists = bool(model_path and Path(model_path).exists())
    if args.require_weights and not model_exists:
        errors.append('model path not found: ' + model_path)

    report = {
        'status': 'ok' if not errors else 'failed',
        'config': args.config,
        'cuda_available': cuda_available,
        'device_count': device_count,
        'device_name': device_name,
        'memory_bytes': memory_bytes,
        'model_path': model_path,
        'model_exists': model_exists,
        'quantize': pcfg.get('quantize'),
        'errors': errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
