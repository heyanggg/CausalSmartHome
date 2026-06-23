"""Compatibility shim for loading NumPy-2.x pickles in NumPy-1.x envs.

Some SmartGen/SmartGuard pkl files were saved with module paths like
numpy._core.*, while older NumPy exposes them as numpy.core.*.
Python imports sitecustomize automatically when PYTHONPATH=. is used.
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
