#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

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
from causal_smart_home.schema import BehaviorSequence
from causal_smart_home.target_distribution_guard import compute_device_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify generated sequences and write repair prompts; no external API is called.")
    parser.add_argument("--generated-pkl", required=True)
    parser.add_argument("--guarded-hints-json", required=True)
    parser.add_argument("--guard-report-json", required=True)
    parser.add_argument("--target-pkl")
    parser.add_argument("--target-distribution-json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max-copies", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated_pkl = Path(args.generated_pkl).resolve()
    hints_json = Path(args.guarded_hints_json).resolve()
    guard_json = Path(args.guard_report_json).resolve()
    if not generated_pkl.exists():
        raise FileNotFoundError(f"--generated-pkl not found: {generated_pkl}")
    if not hints_json.exists():
        raise FileNotFoundError(f"--guarded-hints-json not found: {hints_json}")
    if not guard_json.exists():
        raise FileNotFoundError(f"--guard-report-json not found: {guard_json}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sequences = load_pickle_sequences(generated_pkl)
    hints_payload = json.loads(hints_json.read_text(encoding="utf-8"))
    guard_report = json.loads(guard_json.read_text(encoding="utf-8"))
    guarded_edges = extract_guarded_edges(hints_payload)
    target_distribution = load_target_distribution(args)

    scores = score_sequences_causal_tof(
        sequences,
        guarded_edges,
        target_distribution=target_distribution,
        mode="weight",
        temperature=args.temperature,
    )
    overused_devices = guard_report.get("overused_devices", [])

    repair_rows: list[dict[str, Any]] = []
    with open(out_dir / "repair_prompts.jsonl", "w", encoding="utf-8") as f:
        for seq, score in zip(sequences, scores):
            if score.get("violated_edges") or score.get("distribution_penalty", 0.0) > 0:
                prompt = build_repair_prompt(seq, score, overused_devices, target_distribution_warning=target_distribution)
                row = {"index": score.get("index"), "prompt": prompt, "violated_edges": score.get("violated_edges", [])}
                repair_rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    resampled, resampling_config = weighted_resample_sequences(sequences, scores, seed=args.seed, max_copies=args.max_copies)
    save_pickle_sequences(out_dir / "kept_or_weighted_sequences.pkl", resampled)
    (out_dir / "verification_scores.json").write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "weights.json").write_text(
        json.dumps({"weights": [float(score["sample_weight"]) for score in scores], "mode": "weight"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "resampling_config.json").write_text(json.dumps(resampling_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"verified sequences: {len(sequences)}")
    print(f"repair prompts written: {len(repair_rows)}")
    print(f"saved outputs under: {out_dir}")


def build_repair_prompt(
    sequence: BehaviorSequence,
    score: Mapping[str, Any],
    overused_devices: list[Mapping[str, Any]] | list[str],
    target_distribution_warning: Mapping[str, float] | None = None,
) -> str:
    overused_text = json.dumps(overused_devices, ensure_ascii=False, indent=2)
    violated_text = json.dumps(score.get("violated_edges", []), ensure_ascii=False, indent=2)
    target_warning_text = json.dumps(target_distribution_warning or {}, ensure_ascii=False, indent=2)
    original_sequence = sequence.to_flat_numeric()
    return "\n".join(
        [
            "Repair this SmartGen generated smart-home behavior sequence with minimal changes.",
            "Do not call external tools. Do not invent devices or actions outside the legal SmartGen/SmartGuard numeric format.",
            "",
            "Original flattened sequence:",
            json.dumps(original_sequence, ensure_ascii=False),
            "",
            "Violated causal edges that should be repaired if possible:",
            violated_text,
            "",
            "Overused devices from the target-distribution guard report:",
            overused_text,
            "",
            "Target context distribution warning:",
            target_warning_text,
            "",
            "Instructions:",
            "1. Make the smallest possible edit to ordering, deletion, or substitution.",
            "2. Preserve the target context and legal device/action ids.",
            "3. Prefer satisfying guarded causal-reweighted GSS hints over raw GCAD hints.",
            "4. Return only the repaired flattened list in the same format.",
        ]
    )


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
