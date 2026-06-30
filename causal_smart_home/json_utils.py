"""JSON 输出辅助函数。

实验脚本会频繁写 Path、NumPy 标量、NaN 等对象到 JSON。集中放在这里可以避免
每个脚本各自复制一份 ``jsonable``。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


def jsonable(obj: Any) -> Any:
    """把常见 Python/NumPy/Path 对象转换成稳定可写的 JSON 值。"""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if hasattr(obj, "item"):
        try:
            return jsonable(obj.item())
        except Exception:
            return str(obj)
    return obj


def write_json(path: str | Path, payload: Mapping[str, Any] | list[Any]) -> None:
    """写出缩进格式 JSON，并自动创建父目录。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
