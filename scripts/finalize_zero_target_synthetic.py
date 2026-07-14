#!/usr/bin/env python
"""Lock a zero-target synthetic corpus before Gen original TOF.

Baseline mode preserves the source-only generated corpus. Causal-GSS mode
performs deterministic source-causal weighted finalization using only the
already generated sequences and source-derived GSS hints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.causal.refinement.causal_tof import (
    extract_guarded_edges,
    load_pickle_sequences,
    save_pickle_sequences,
    score_sequences_causal_tof,
    weighted_resample_sequences,
)
from causal_smart_home.json_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pkl", required=True, type=Path)
    parser.add_argument("--out-pkl", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=["zero_target_baseline", "zero_target_causal_gss"])
    parser.add_argument("--source-causal-hints-json", type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out-report", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences = load_pickle_sequences(args.input_pkl)
    if args.variant == "zero_target_baseline":
        if args.source_causal_hints_json is not None:
            raise ValueError("zero_target_baseline must not receive causal hints")
        selected = sequences
        consistency_summary = None
        method = "source_only_corpus_lock"
    else:
        if args.source_causal_hints_json is None:
            raise ValueError("zero_target_causal_gss requires source causal hints")
        hints = json.loads(args.source_causal_hints_json.read_text(encoding="utf-8"))
        if hints.get("target_data_used") is not False or hints.get("adaptation_mode") != "source_only":
            raise ValueError("zero_target_causal_gss requires audited source-only hints")
        scores = score_sequences_causal_tof(
            sequences,
            extract_guarded_edges(hints),
            target_distribution=None,
            mode="weight",
            alpha_rec=0.0,
            beta_inconsistency=1.0,
            gamma_dist=0.0,
            temperature=1.0,
        )
        selected, sampling = weighted_resample_sequences(
            sequences, scores, seed=args.seed, max_copies=3, target_size=len(sequences)
        )
        values = [float(score["causal_consistency_score"]) for score in scores]
        consistency_summary = {
            "mean": sum(values) / len(values) if values else 0.0,
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
            "sampling": sampling,
        }
        method = "source_only_causal_gss_weighted_synthesis_finalization"
    save_pickle_sequences(args.out_pkl, selected)
    write_json(
        args.out_report,
        {
            "schema_version": 1,
            "variant": args.variant,
            "seed": args.seed,
            "method": method,
            "input_pkl": str(args.input_pkl.resolve()),
            "output_pkl": str(args.out_pkl.resolve()),
            "input_size": len(sequences),
            "output_size": len(selected),
            "target_data_used": False,
            "source_causal_hints_json": str(args.source_causal_hints_json.resolve()) if args.source_causal_hints_json else None,
            "causal_consistency_summary": consistency_summary,
            "synthetic_data_locked": True,
        },
    )


if __name__ == "__main__":
    main()
