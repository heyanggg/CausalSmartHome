from __future__ import annotations

from pathlib import Path
import json
import subprocess
from typing import Sequence

from .schema import BehaviorSequence
from .smartgen_adapter import SmartGenAdapter


class SmartGuardAdapter:
    """Subprocess wrapper for unmodified SmartGuard scripts."""

    def __init__(self, smartguard_root: str) -> None:
        self.smartguard_root = Path(smartguard_root).resolve()

    def write_sequences(self, relative_path: str, sequences: Sequence[BehaviorSequence]) -> Path:
        path = self.smartguard_root / relative_path
        SmartGenAdapter.save_pkl_sequences(path, sequences)
        return path

    def train(self, dataset: str, extra_args: list[str] | None = None, timeout: int | None = None) -> subprocess.CompletedProcess:
        cmd = ["python", "train.py", "--dataset", dataset, "--model", "SmartGuard"]
        if extra_args:
            cmd.extend(extra_args)
        return subprocess.run(cmd, cwd=self.smartguard_root, check=True, text=True, capture_output=True, timeout=timeout)

    def evaluate(self, dataset: str, model_path: str | None = None, output: str | None = None, extra_args: list[str] | None = None, timeout: int | None = None) -> subprocess.CompletedProcess:
        script = self.smartguard_root / "evaluate_smartguard.py"
        if not script.exists():
            raise FileNotFoundError("evaluate_smartguard.py is not present; use the adapter package copy or add the wrapper script")
        cmd = ["python", str(script.name), "--dataset", dataset]
        if model_path:
            cmd += ["--model_path", model_path]
        if output:
            cmd += ["--output", output]
        if extra_args:
            cmd.extend(extra_args)
        return subprocess.run(cmd, cwd=self.smartguard_root, check=True, text=True, capture_output=True, timeout=timeout)
