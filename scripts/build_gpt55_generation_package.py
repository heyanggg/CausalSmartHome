#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a GPT-5.5 generation package from a causal GSS artifact directory.")
    parser.add_argument("--causal-gss-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scenario", required=True, choices=["fr_st", "sp_st"])
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    causal_gss_dir = Path(args.causal_gss_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not causal_gss_dir.exists():
        raise FileNotFoundError(f"--causal-gss-dir not found: {causal_gss_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    required = [
        "prompt.txt",
        "guard_report.json",
        "guarded_reweighted_gss_hints.json",
        "resolved_causal_relation_prior.json",
    ]
    for name in required:
        src = causal_gss_dir / name
        if not src.exists():
            raise FileNotFoundError(f"required causal GSS artifact missing: {src}")
        shutil.copyfile(src, out_dir / name)

    schema = {
        "generator": "gpt55_generation",
        "api_llm": False,
        "manual_generation": True,
        "scenario": args.scenario,
        "seed": args.seed,
        "sequence_format": "flat_quadruples",
        "fields": ["day", "hour_slot", "device_id", "action_id"],
        "causal_gss_dir": str(causal_gss_dir),
        "guard_mode": "downweight",
        "downweight_factor": 0.25,
        "reweight_mode": "multiplicative",
        "lambda_causal": 1.0,
    }
    (out_dir / "generation_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "generation_instruction.md").write_text(build_instruction(args.scenario), encoding="utf-8")
    print(f"saved GPT-5.5 generation package: {out_dir}")


def build_instruction(scenario: str) -> str:
    scenario_note = "In SP-ST, pay special attention to Television overuse." if scenario == "sp_st" else "Use FR-ST target-context normal behavior."
    return "\n".join(
        [
            "# GPT-5.5 Generation Instruction",
            "",
            "Use GPT-5.5 as the Gen target-context behavior generator for this project.",
            "",
            "Use:",
            "1. prompt.txt",
            "2. guarded_reweighted_gss_hints.json",
            "3. guard_report.json",
            "4. resolved_causal_relation_prior.json",
            "",
            "Generate target-context normal smart-home behavior sequences.",
            "",
            "Output format:",
            "Each sequence must be a flat quadruple list:",
            "[day, hour_slot, device_id, action_id, day, hour_slot, device_id, action_id, ...]",
            "",
            "Keep the same sequence length convention as the generated pkl files in this project.",
            "If the reference files use 10 events per sequence, output 40 integers per sequence.",
            "",
            "Follow these rules:",
            "1. Use guarded causal-reweighted GSS hints as primary structure.",
            "2. Use raw causal relation hints only as weak background.",
            "3. If raw causal relation conflicts with guarded hints, follow guarded hints.",
            "4. Do not over-generate devices listed as overused in guard_report.",
            f"5. {scenario_note}",
            "6. Keep target-context device/action distribution plausible.",
            "7. Do not generate device/action IDs outside known dictionaries.",
            "8. Preserve reasonable temporal order.",
            "9. Generate normal behavior sequences, not attacks.",
            "10. Save metadata with generator = gpt55_generation and api_llm = false.",
            "",
            "This package is the fixed GPT-5.5 generation protocol for CausalSmartHome main experiments.",
            "",
        ]
    )


if __name__ == "__main__":
    main()
