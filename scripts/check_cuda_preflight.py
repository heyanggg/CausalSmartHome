#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from typing import Any


def collect() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "torch_importable": False,
            "torch_version": None,
            "cuda_available": False,
            "cuda_device_count": 0,
            "device_name": None,
            "current_device": None,
            "cudnn_enabled": None,
            "error": str(exc),
        }

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if hasattr(torch.cuda, "device_count") else 0
    current_device = None
    device_name = None
    if cuda_available and device_count:
        current_device = int(torch.cuda.current_device())
        device_name = torch.cuda.get_device_name(current_device)
    return {
        "torch_importable": True,
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "device_name": device_name,
        "current_device": current_device,
        "cudnn_enabled": bool(getattr(torch.backends, "cudnn", None) and torch.backends.cudnn.enabled),
    }


def main() -> None:
    payload = collect()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["cuda_available"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
