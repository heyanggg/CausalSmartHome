#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Codex/GPT-5.5 surrogate generation package from a Stage4A directory.")
    parser.add_argument("--stage4a-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scenario", required=True, choices=["fr_st", "sp_st"])
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage4a_dir = Path(args.stage4a_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not stage4a_dir.exists():
        raise FileNotFoundError(f"--stage4a-dir not found: {stage4a_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    required = [
        "prompt.txt",
        "guard_report.json",
        "guarded_reweighted_gss_hints.json",
        "resolved_gcad_prior.json",
    ]
    for name in required:
        src = stage4a_dir / name
        if not src.exists():
            raise FileNotFoundError(f"required Stage4A artifact missing: {src}")
        shutil.copyfile(src, out_dir / name)

    schema = {
        "generator": "codex_gpt55_surrogate",
        "api_llm": False,
        "surrogate_for_smartgen_llm": True,
        "scenario": args.scenario,
        "seed": args.seed,
        "sequence_format": "flat_quadruples",
        "fields": ["day", "hour_slot", "device_id", "action_id"],
        "stage4a_dir": str(stage4a_dir),
        "guard_mode": "downweight",
        "downweight_factor": 0.25,
        "reweight_mode": "multiplicative",
        "lambda_causal": 1.0,
    }
    (out_dir / "generation_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "generation_instruction.md").write_text(build_instruction(args.scenario), encoding="utf-8")
    print(f"saved Codex/GPT-5.5 generation package: {out_dir}")


def build_instruction(scenario: str) -> str:
    scenario_note = "In SP-ST, pay special attention to Television overuse." if scenario == "sp_st" else "Use FR-ST target-context normal behavior."
    return "\n".join(
        [
            "# Codex/GPT-5.5 Surrogate Generation Instruction",
            "",
            "You are Codex/GPT-5.5 acting as the SmartGen LLM surrogate generator for this project.",
            "",
            "Use:",
            "1. prompt.txt",
            "2. guarded_reweighted_gss_hints.json",
            "3. guard_report.json",
            "4. resolved_gcad_prior.json",
            "",
            "Generate target-context normal smart-home behavior sequences.",
            "",
            "Output format:",
            "Each sequence must be a flat quadruple list:",
            "[day, hour_slot, device_id, action_id, day, hour_slot, device_id, action_id, ...]",
            "",
            "Keep the same sequence length convention as existing Stage3 generated pkl.",
            "If old Stage3 uses 10 events per sequence, output 40 integers per sequence.",
            "",
            "Follow these rules:",
            "1. Use guarded causal-reweighted GSS hints as primary structure.",
            "2. Use raw GCAD hints only as weak background.",
            "3. If raw GCAD conflicts with guarded hints, follow guarded hints.",
            "4. Do not over-generate devices listed as overused in guard_report.",
            f"5. {scenario_note}",
            "6. Keep target-context device/action distribution plausible.",
            "7. Do not generate device/action IDs outside known dictionaries.",
            "8. Preserve reasonable temporal order.",
            "9. Generate normal behavior sequences, not attacks.",
            "10. Save metadata with generator = codex_gpt55_surrogate and api_llm = false.",
            "",
            "This package is not a SmartGen paper API LLM reproduction. It is the fixed surrogate-generation protocol for CausalSmartHome Stage4.",
            "",
        ]
    )


if __name__ == "__main__":
    main()
