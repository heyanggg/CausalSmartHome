from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def lineage_env(tmp_path: Path, target: Path, *, allowed: bool) -> tuple[dict[str, str], Path]:
    out = tmp_path / ("allowed.json" if allowed else "blocked.json")
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "CSH_LINEAGE_OUT": str(out),
            "CSH_LINEAGE_STAGE": "test_stage",
            "CSH_LINEAGE_VARIANT": "zero_target_full",
            "CSH_LINEAGE_DATASET": "sp",
            "CSH_LINEAGE_SCENARIO": "st",
            "CSH_LINEAGE_SEED": "2024",
            "CSH_LINEAGE_PURPOSE": "offline_generation_quality_evaluation" if allowed else "filtering",
            "CSH_TARGET_NORMAL_FILES": json.dumps([str(target.resolve())]),
            "CSH_TARGET_ATTACK_FILES": "[]",
            "CSH_ALLOW_TARGET_NORMAL": "1" if allowed else "0",
            "CSH_ALLOW_TARGET_ATTACK": "0",
        }
    )
    return env, out


def test_actual_open_gate_blocks_target_normal(tmp_path: Path) -> None:
    target = tmp_path / "target_normal.pkl"
    target.write_bytes(b"not important")
    env, out = lineage_env(tmp_path, target, allowed=False)
    result = subprocess.run(
        [sys.executable, "-c", f"open({str(target)!r}, 'rb').read()"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "zero-target gate blocked target normal read" in result.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["target_normal_consumed"] is True
    assert payload["status"] == "blocked_target_normal_access"


def test_evaluation_only_actual_open_is_recorded(tmp_path: Path) -> None:
    target = tmp_path / "target_normal.pkl"
    target.write_bytes(b"ok")
    env, out = lineage_env(tmp_path, target, allowed=True)
    result = subprocess.run(
        [sys.executable, "-c", f"open({str(target)!r}, 'rb').read()"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["target_normal_consumed"] is True
    assert payload["purpose"] == "offline_generation_quality_evaluation"


def test_zero_target_causal_tof_requires_disabled_distribution_penalty() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_causal_tof.py",
            "--generated-pkl", "missing.pkl",
            "--guarded-hints-json", "missing.json",
            "--method-line", "zero_target",
            "--out-scores", "scores.json",
            "--out-weights", "weights.json",
            "--out-weighted-resampled-pkl", "out.pkl",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "requires gamma_dist=0" in result.stderr
