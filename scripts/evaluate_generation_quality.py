#!/usr/bin/env python
"""Compute generation quality and export paper case-study matrices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.causal.evaluation.generation_quality import evaluate_generation_quality_files
from causal_smart_home.experiment_paths import VARIANTS
from causal_smart_home.gen_downstream_ad import DATASETS, ENV_BY_SCENARIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-pkl", required=True, type=Path)
    parser.add_argument("--synthetic-pkl", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--scenario", required=True, choices=sorted(ENV_BY_SCENARIO))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_generation_quality_files(
        args.target_pkl,
        args.synthetic_pkl,
        args.out_dir,
        dataset=args.dataset,
        scenario=args.scenario,
        seed=args.seed,
        variant=args.variant,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
