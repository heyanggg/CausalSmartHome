#!/usr/bin/env python
"""打包供 Codex 生成 target-normal 行为序列使用的 prompt 产物。

本脚本本身不生成序列，只把固定的 causal-GSS 产物复制成一个自包含 package，
并写入 schema/instruction 文件，保证人工或 Codex 生成步骤可以复现。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Codex generation package from a causal GSS artifact directory.")
    parser.add_argument("--causal-gss-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scenario", required=True, help="Scenario key such as fr_st, sp_st, or us_nt.")
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
        "causal_reweighted_gss_hints.json",
        "resolved_causal_relation_prior.json",
    ]
    for name in required:
        # 保持 generation package 自包含：后续验证脚本只需要指向这个目录，不必
        # 再依赖原始 causal-GSS 运行目录是否还存在。
        src = causal_gss_dir / name
        if not src.exists():
            raise FileNotFoundError(f"required causal GSS artifact missing: {src}")
        shutil.copyfile(src, out_dir / name)

    schema = {
        "generator": "codex_generation",
        "generation_model": "Codex",
        "manual_generation": True,
        "scenario": args.scenario,
        "seed": args.seed,
        "sequence_format": "flat_quadruples",
        "fields": ["day", "hour_slot", "device_id", "action_id"],
        "causal_gss_dir": str(causal_gss_dir),
        "target_data_used": False,
        "reweight_mode": "multiplicative",
        "lambda_causal": 1.0,
    }
    (out_dir / "generation_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "generation_instruction.md").write_text(build_instruction(args.scenario), encoding="utf-8")
    print(f"saved Codex generation package: {out_dir}")


def build_instruction(scenario: str) -> str:
    """为单个场景 package 写出面向人工/Codex 的生成协议说明。"""
    suffix = scenario.rsplit("_", 1)[-1]
    scenario_note = {
        "st": "Adapt from winter to spring using general seasonal knowledge.",
        "tt": "Adapt from daytime activity to nighttime activity using the declared schedule change.",
        "nt": "Adapt from single occupancy to multiple occupancy using the declared household change.",
    }.get(suffix, f"Adapt behavior to the declared {scenario} context transition.")
    return "\n".join(
        [
            "# Codex Generation Instruction",
            "",
            "Use Codex as the target-context behavior generator for this project.",
            "",
            "Use:",
            "1. prompt.txt",
            "2. causal_reweighted_gss_hints.json",
            "3. resolved_causal_relation_prior.json",
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
            "1. Use causal-reweighted GSS hints as primary structure.",
            "2. Use raw causal relation hints only as weak background.",
            "3. If raw causal relation conflicts with reweighted hints, follow reweighted hints.",
            f"4. {scenario_note}",
            "5. Use only the declared context transition; no target behavior samples are available.",
            "6. Do not generate device/action IDs outside known dictionaries.",
            "7. Preserve reasonable temporal order.",
            "8. Generate normal behavior sequences, not attacks.",
            "9. Save metadata with generator = codex_generation and generation_model = Codex.",
            "",
            "This package is the fixed Codex generation protocol for CausalSmartHome main experiments.",
            "",
        ]
    )


if __name__ == "__main__":
    main()
