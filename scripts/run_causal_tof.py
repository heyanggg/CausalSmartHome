#!/usr/bin/env python
"""Apply causal-consistency refinement after an existing Gen TOF output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.causal.adaptation.target_guard import compute_device_distribution
from causal_smart_home.causal.evaluation.causal_metrics import summarize_causal_consistency
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
    parser.add_argument("--generated-pkl", required=True, type=Path)
    parser.add_argument("--guarded-hints-json", required=True, type=Path)
    parser.add_argument("--target-pkl", type=Path)
    parser.add_argument("--target-distribution-json", type=Path)
    parser.add_argument("--method-line", choices=["zero_target", "target_assisted"], default="target_assisted")
    parser.add_argument("--reconstruction-losses-json", type=Path)
    parser.add_argument("--out-scores", required=True, type=Path)
    parser.add_argument("--out-weights", required=True, type=Path)
    parser.add_argument("--out-weighted-resampled-pkl", required=True, type=Path)
    parser.add_argument("--out-resampling-config", type=Path)
    parser.add_argument("--mode", choices=["rank", "weight", "filter"], default="weight")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha-rec", type=float, default=1.0)
    parser.add_argument("--beta-inconsistency", type=float, default=1.0)
    parser.add_argument("--beta-violation", type=float, help="Deprecated alias for --beta-inconsistency.")
    parser.add_argument("--gamma-dist", type=float, default=1.0)
    parser.add_argument("--penalize-downweighted-edges", action="store_true")
    parser.add_argument("--min-weight", type=float, default=0.05)
    parser.add_argument("--max-copies", type=int, default=3)
    parser.add_argument("--resample-size", type=int)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--input-stage", default="gen_original_tof", choices=["gen_original_tof"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.method_line == "zero_target":
        if args.target_pkl or args.target_distribution_json:
            raise ValueError("zero_target Causal-TOF forbids target normal data")
        if args.gamma_dist != 0:
            raise ValueError("zero_target Causal-TOF requires gamma_dist=0")
    for label, path in (("--generated-pkl", args.generated_pkl), ("--guarded-hints-json", args.guarded_hints_json)):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    sequences = load_pickle_sequences(args.generated_pkl)
    hints = json.loads(args.guarded_hints_json.read_text(encoding="utf-8"))
    edges = extract_guarded_edges(hints)
    target_distribution = load_target_distribution(args)
    reconstruction_losses = load_reconstruction_losses(args)
    beta = args.beta_inconsistency if args.beta_violation is None else args.beta_violation
    scores = score_sequences_causal_tof(
        sequences,
        edges,
        target_distribution=target_distribution,
        reconstruction_losses=reconstruction_losses,
        mode=args.mode,
        min_weight=args.min_weight,
        alpha_rec=args.alpha_rec,
        beta_inconsistency=beta,
        gamma_dist=args.gamma_dist,
        temperature=args.temperature,
        penalize_downweighted_edges=args.penalize_downweighted_edges,
    )

    out_config = args.out_resampling_config or args.out_weighted_resampled_pkl.with_suffix(
        args.out_weighted_resampled_pkl.suffix + ".config.json"
    )
    write_json(args.out_scores, scores)
    write_json(
        args.out_weights,
        {
            "mode": args.mode,
            "score_direction": "higher_is_better",
            "weights": [score["sample_weight"] for score in scores],
            "causal_consistency_summary": summarize_causal_consistency(scores),
            "scores_path": str(args.out_scores.resolve()),
        },
    )

    if args.mode == "filter":
        selected = [sequence for sequence, score in zip(sequences, scores) if score["decision"] == "keep"]
        config = {"mode": "filter", "kept": len(selected), "raw": len(sequences)}
    else:
        selected, config = weighted_resample_sequences(
            sequences,
            scores,
            seed=args.seed,
            max_copies=args.max_copies,
            target_size=args.resample_size,
        )
        config["mode"] = args.mode
    save_pickle_sequences(args.out_weighted_resampled_pkl, selected)
    config.update(
        {
            "seed": args.seed,
            "temperature": args.temperature,
            "alpha_rec": args.alpha_rec,
            "beta_inconsistency": beta,
            "gamma_dist": args.gamma_dist,
            "method_line": args.method_line,
            "distribution_penalty_source": "disabled_zero_target" if args.method_line == "zero_target" else "target_empirical_distribution",
            "min_weight": args.min_weight,
            "max_copies": args.max_copies,
            "resample_size": args.resample_size,
            "input_stage": args.input_stage,
            "used_gen_original_tof": True,
            "used_causal_tof": True,
            "num_generated_after_gen_tof": len(sequences),
            "num_generated_after_causal_tof": len(selected),
            "causal_consistency_summary": summarize_causal_consistency(scores),
        }
    )
    write_json(out_config, config)
    print(json.dumps(config, ensure_ascii=False, indent=2))


def load_target_distribution(args: argparse.Namespace) -> dict[str, float] | None:
    if args.target_distribution_json:
        return {
            str(key): float(value)
            for key, value in json.loads(args.target_distribution_json.read_text(encoding="utf-8")).items()
        }
    if args.target_pkl:
        if not args.target_pkl.exists():
            raise FileNotFoundError(f"--target-pkl not found: {args.target_pkl}")
        return compute_device_distribution(load_pickle_sequences(args.target_pkl))
    return None


def load_reconstruction_losses(args: argparse.Namespace) -> list[float] | None:
    if not args.reconstruction_losses_json:
        return None
    payload = json.loads(args.reconstruction_losses_json.read_text(encoding="utf-8"))
    values = payload.get("reconstruction_losses", payload) if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError("reconstruction losses JSON must be a list or contain reconstruction_losses")
    return [float(value) for value in values]


if __name__ == "__main__":
    main()
