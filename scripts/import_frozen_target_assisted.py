#!/usr/bin/env python
"""Import the frozen SP-ST target-assisted upper bound without modifying it."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pkl", required=True, type=Path)
    parser.add_argument("--out-pkl", required=True, type=Path)
    parser.add_argument("--out-report", required=True, type=Path)
    parser.add_argument("--historical-config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    historical = json.loads(args.historical_config.read_text(encoding="utf-8"))
    if historical.get("target_data_used") is not True or not historical.get("target_pkl"):
        raise ValueError("frozen artifact is not documented as target-assisted")
    with args.input_pkl.open("rb") as handle:
        sequences = pickle.load(handle)
    args.out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_pkl.open("wb") as handle:
        pickle.dump(sequences, handle)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "variant": "target_assisted_full",
                "method_class": "target_assisted_upper_bound_non_zero_target",
                "seed": args.seed,
                "input_pkl": str(args.input_pkl.resolve()),
                "output_pkl": str(args.out_pkl.resolve()),
                "synthetic_data_locked": True,
                "runtime_target_normal_read": False,
                "inherited_target_normal_dependency": True,
                "historical_target_normal_file": str(Path(historical["target_pkl"]).resolve()),
                "historical_config": str(args.historical_config.resolve()),
                "num_sequences": len(sequences),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
