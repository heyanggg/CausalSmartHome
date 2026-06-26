#!/usr/bin/env python
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT / "outputs" / "main_experiment" / "summary" / "main_experiment_aggregate.json"
EXPECTED = {
    "ablation_no_causal_tof": {
        "precision_mean": 0.8400260212783585,
        "recall_mean": 0.9922161172161172,
        "f1_mean": 0.8981913177582828,
        "fpr_mean": 0.2564102564102564,
    },
    "proposed_causal_gss_gpt55_causal_tof": {
        "precision_mean": 0.9538611152318317,
        "recall_mean": 0.9977106227106227,
        "f1_mean": 0.9752202755560685,
        "fpr_mean": 0.048534798534798536,
    },
}


def assert_close(name: str, got: float, expected: float, tol: float = 1e-9) -> None:
    if not math.isfinite(float(got)) or abs(float(got) - expected) > tol:
        raise AssertionError(f"{name}: got {got!r}, expected {expected!r}")


def main() -> None:
    if not AGG.exists():
        raise FileNotFoundError(f"missing aggregate summary: {AGG}")
    payload = json.loads(AGG.read_text(encoding="utf-8"))
    rows = {row["variant"]: row for row in payload.get("rows", [])}
    missing = sorted(set(EXPECTED) - set(rows))
    if missing:
        raise AssertionError(f"missing variants in aggregate summary: {missing}")
    for variant, expected_metrics in EXPECTED.items():
        row = rows[variant]
        for metric, expected_value in expected_metrics.items():
            assert_close(f"{variant}.{metric}", row[metric], expected_value)
    frozen = ROOT / "outputs" / "main_experiment_frozen" / "sp_st_gpt55_proposed_3seed_20260623"
    required = [
        frozen / "generated" / "sp_st" / "seed2024" / "generated_gpt55_clean.pkl",
        frozen / "generated" / "sp_st" / "seed2025" / "generated_gpt55_clean.pkl",
        frozen / "generated" / "sp_st" / "seed2026" / "generated_gpt55_clean.pkl",
        frozen / "run_reproduce_from_frozen.sh",
    ]
    missing_files = [str(path) for path in required if not path.exists()]
    if missing_files:
        raise AssertionError("missing frozen reproducibility files: " + "; ".join(missing_files))
    print("RECOVERY_INTEGRITY_OK: SP-ST good metrics and frozen inputs are present.")


if __name__ == "__main__":
    main()
