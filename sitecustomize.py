"""兼容不同 NumPy 版本生成的 pickle。

部分 Gen 的 pkl 文件是在 NumPy 2.x 环境下保存的，pickle 内部会记录
``numpy._core.*`` 这类模块路径；而较老的 NumPy 环境只暴露
``numpy.core.*``。当运行时设置了 ``PYTHONPATH=.``，Python 会自动导入
``sitecustomize``，这里就提前注册别名，避免加载旧实验产物时因为模块路径
不同而失败。
"""
from __future__ import annotations

import sys

try:
    import numpy as _np
    import numpy.core as _np_core

    if not hasattr(_np, "_core"):
        setattr(_np, "_core", _np_core)

    sys.modules.setdefault("numpy._core", _np_core)

    for _name in [
        "multiarray",
        "numeric",
        "fromnumeric",
        "umath",
        "_multiarray_umath",
        "numerictypes",
        "arrayprint",
        "defchararray",
        "records",
        "memmap",
        "overrides",
    ]:
        try:
            _mod = __import__(f"numpy.core.{_name}", fromlist=["*"])
            setattr(_np_core, _name, _mod)
            sys.modules.setdefault(f"numpy._core.{_name}", _mod)
        except Exception:
            pass
except Exception:
    pass
