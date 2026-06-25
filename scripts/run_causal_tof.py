#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.causal_tof import (
    extract_guarded_edges,
    load_pickle_sequences,
    save_pickle_sequences,
    score_sequences_causal_tof,
    weighted_resample_sequences,
)
from causal_smart_home.target_distribution_guard import compute_device_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Stage 4 Causal-TOF soft weighting to generated sequences.")
    parser.add_argument("--generated-pkl", required=True, help="Input pkl. For the corrected mainline this should be SmartGen original TOF output, not the fresh raw generation.")
    parser.add_argument("--guarded-hints-json", required=True)
    parser.add_argument("--target-pkl", help="Target normal pkl for distribution penalty.")
    parser.add_argument("--target-distribution-json", help="Optional precomputed target device distribution JSON.")
    parser.add_argument("--out-scores", required=True)
    parser.add_argument("--out-weights", required=True)
    parser.add_argument("--out-weighted-resampled-pkl", required=True)
    parser.add_argument("--out-resampling-config", help="Defaults to <out-weighted-resampled-pkl>.config.json")
    parser.add_argument("--mode", choices=["rank", "weight", "filter"], default="weight")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha-rec", type=float, default=1.0)
    parser.add_argument("--beta-violation", type=float, default=1.0)
    parser.add_argument("--gamma-dist", type=float, default=1.0)
    parser.add_argument("--min-weight", type=float, default=0.05)
    parser.add_argument("--max-copies", type=int, default=3)
    parser.add_argument("--resample-size", type=int)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--input-stage", default="gen_original_tof", choices=["gen_original_tof"], help="Provenance marker for the input pkl.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated_pkl = Path(args.generated_pkl).resolve()
    hints_path = Path(args.guarded_hints_json).resolve()
    if not generated_pkl.exists():
        raise FileNotFoundError(f"--generated-pkl not found: {generated_pkl}")
    if not hints_path.exists():
        raise FileNotFoundError(f"--guarded-hints-json not found: {hints_path}")

    sequences = load_pickle_sequences(generated_pkl)
    hints_payload = json.loads(hints_path.read_text(encoding="utf-8"))
    guarded_edges = extract_guarded_edges(hints_payload)
    target_distribution = load_target_distribution(args)

    scores = score_sequences_causal_tof(
        sequences,
        guarded_edges,
        target_distribution=target_distribution,
        mode=args.mode,
        min_weight=args.min_weight,
        alpha_rec=args.alpha_rec,
        beta_violation=args.beta_violation,
        gamma_dist=args.gamma_dist,
        temperature=args.temperature,
    )

    out_scores = Path(args.out_scores).resolve()
    out_weights = Path(args.out_weights).resolve()
    out_resampled = Path(args.out_weighted_resampled_pkl).resolve()
    out_config = Path(args.out_resampling_config).resolve() if args.out_resampling_config else out_resampled.with_suffix(out_resampled.suffix + ".config.json")
    for path in (out_scores, out_weights, out_resampled, out_config):
        path.parent.mkdir(parents=True, exist_ok=True)

    out_scores.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    weights = [float(score["sample_weight"]) for score in scores]
    out_weights.write_text(json.dumps({"mode": args.mode, "weights": weights, "scores_path": str(out_scores)}, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.mode == "filter":
        kept = [seq for seq, score in zip(sequences, scores) if score.get("decision") == "keep"]
        resampling_config = {"mode": "filter", "kept": len(kept), "raw": len(sequences), "min_weight": args.min_weight}
        save_pickle_sequences(out_resampled, kept)
    else:
        resampled, resampling_config = weighted_resample_sequences(
            sequences,
            scores,
            seed=args.seed,
            max_copies=args.max_copies,
            target_size=args.resample_size or len(sequences),
        )
        resampling_config["mode"] = args.mode
        save_pickle_sequences(out_resampled, resampled)
    resampling_config["input_stage"] = args.input_stage
    resampling_config["used_smartgen_original_tof"] = args.input_stage == "smartgen_original_tof"
    resampling_config["used_causal_tof"] = True
    resampling_config["num_generated_after_smartgen_tof"] = len(sequences) if args.input_stage == "smartgen_original_tof" else None
    resampling_config["num_generated_after_causal_tof"] = len(resampled) if args.mode != "filter" else len(kept)
    out_config.write_text(json.dumps(resampling_config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"scored sequences: {len(scores)}")
    print(f"saved scores: {out_scores}")
    print(f"saved weights: {out_weights}")
    print(f"saved weighted/resampled pkl: {out_resampled}")
    print(f"saved resampling config: {out_config}")


def load_target_distribution(args: argparse.Namespace) -> dict[str, float] | None:
    if args.target_distribution_json:
        path = Path(args.target_distribution_json).resolve()
        if not path.exists():
            raise FileNotFoundError(f"--target-distribution-json not found: {path}")
        return {str(k): float(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    if args.target_pkl:
        path = Path(args.target_pkl).resolve()
        if not path.exists():
            raise FileNotFoundError(f"--target-pkl not found: {path}")
        return compute_device_distribution(load_pickle_sequences(path))
    return None


if __name__ == "__main__":
    main()
