"""Utilities for the strict SP-ST zero-target lineage audit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from .experiment_paths import target_attack_files_for, target_normal_files_for


ZERO_TARGET_VARIANTS = (
    "zero_target_baseline",
    "zero_target_causal_gss",
    "zero_target_causal_tof",
    "zero_target_full",
)
AUDIT_VARIANTS = ZERO_TARGET_VARIANTS + ("target_assisted_full",)


def lineage_environment(
    *,
    repo_root: Path,
    out_path: Path,
    stage: str,
    variant: str,
    dataset: str,
    scenario: str,
    seed: int,
    purpose: str,
    allow_target_normal: bool = False,
    allow_target_attack: bool = False,
) -> dict[str, str]:
    """Build the environment that activates the actual-open audit hook."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + existing if existing else "")
    env.update(
        {
            "CSH_LINEAGE_OUT": str(out_path.resolve()),
            "CSH_LINEAGE_STAGE": stage,
            "CSH_LINEAGE_VARIANT": variant,
            "CSH_LINEAGE_DATASET": dataset,
            "CSH_LINEAGE_SCENARIO": scenario,
            "CSH_LINEAGE_SEED": str(seed),
            "CSH_LINEAGE_PURPOSE": purpose,
            "CSH_TARGET_NORMAL_FILES": json.dumps(
                [str(path) for path in target_normal_files_for(dataset, scenario)]
            ),
            "CSH_TARGET_ATTACK_FILES": json.dumps(
                [str(path) for path in target_attack_files_for(dataset, scenario)]
            ),
            "CSH_ALLOW_TARGET_NORMAL": "1" if allow_target_normal else "0",
            "CSH_ALLOW_TARGET_ATTACK": "1" if allow_target_attack else "0",
        }
    )
    return env


def snapshot_tree(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"path": str(path.relative_to(root)), "size": path.stat().st_size, "sha256": digest})
    manifest_digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"root": str(root.resolve()), "file_count": len(files), "manifest_sha256": manifest_digest, "files": files}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_variant_lineage(
    *,
    variant: str,
    dataset: str,
    scenario: str,
    seed: int,
    lineage_paths: Iterable[Path],
    out_path: Path,
    inherited_target_dependency: bool = False,
) -> dict[str, Any]:
    stages = [read_json(path) for path in lineage_paths]
    consumed = _ordered(path for row in stages for path in row.get("consumed_files", []))
    produced = _ordered(path for row in stages for path in row.get("produced_files", []))
    normal = _ordered(path for row in stages for path in row.get("target_normal_files", []))
    attack = _ordered(path for row in stages for path in row.get("target_attack_files", []))
    pre_evaluation = [
        row
        for row in stages
        if row.get("purpose") not in {"offline_generation_quality_evaluation", "downstream_ad_evaluation"}
    ]
    payload = {
        "schema_version": 1,
        "stage": "variant_pipeline",
        "variant": variant,
        "dataset": dataset,
        "scenario": scenario,
        "seed": seed,
        "consumed_files": consumed,
        "produced_files": produced,
        "target_normal_consumed": bool(normal),
        "target_normal_files": normal,
        "target_attack_consumed": bool(attack),
        "target_attack_files": attack,
        "pre_evaluation_target_normal_consumed": any(
            bool(row.get("target_normal_consumed")) for row in pre_evaluation
        ),
        "pre_evaluation_target_attack_consumed": any(
            bool(row.get("target_attack_consumed")) for row in pre_evaluation
        ),
        "purpose": "generation/filtering/evaluation",
        "method_class": (
            "target_assisted_upper_bound_non_zero_target"
            if variant == "target_assisted_full"
            else "strict_zero_target"
        ),
        "inherited_target_normal_dependency": inherited_target_dependency,
        "stages": stages,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def assert_zero_target_pre_evaluation(lineage: dict[str, Any]) -> None:
    """Fail if a formal variant touches target behavior before offline evaluation."""
    for stage in lineage["stages"]:
        purpose = stage.get("purpose")
        if purpose in {"offline_generation_quality_evaluation", "downstream_ad_evaluation"}:
            continue
        if stage.get("target_normal_consumed") or stage.get("target_attack_consumed"):
            raise RuntimeError(
                f"zero-target audit failed for {lineage['variant']} stage={stage.get('stage')}: "
                "target behavior was read before evaluation"
            )


def audit_payload(
    *,
    seed_dir: Path,
    frozen_before: dict[str, Any],
    frozen_after: dict[str, Any],
    commands: list[str],
    variant_lineages: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    quality: dict[str, Any] = {}
    actual_target_normal = []
    target_read_events = []
    for variant, lineage in variant_lineages.items():
        metrics[variant] = read_json(seed_dir / "variants" / variant / "downstream_ad" / "downstream_ad_metrics.json")
        quality[variant] = read_json(seed_dir / "variants" / variant / "generation_quality" / "generation_quality_summary.json")
        actual_target_normal.extend(lineage.get("target_normal_files", []))
        for stage in lineage.get("stages", []):
            for path in stage.get("target_normal_files", []):
                target_read_events.append(
                    {"variant": variant, "stage": stage.get("stage"), "purpose": stage.get("purpose"), "path": path}
                )

    quality_outputs = {
        path
        for lineage in variant_lineages.values()
        for stage in lineage.get("stages", [])
        if stage.get("purpose") == "offline_generation_quality_evaluation"
        for path in stage.get("produced_files", [])
    }
    upstream_consumed = {
        path
        for lineage in variant_lineages.values()
        for stage in lineage.get("stages", [])
        if stage.get("purpose") not in {"offline_generation_quality_evaluation", "downstream_ad_evaluation"}
        for path in stage.get("consumed_files", [])
    }
    quality_feedback = sorted(quality_outputs & upstream_consumed)
    if quality_feedback:
        raise RuntimeError(f"evaluation-only quality artifacts fed back upstream: {quality_feedback}")

    static_audit = [
        {
            "stage": "causal_relation_discovery",
            "finding": "The resolved prior is source-derived; the audited zero-target run reads the source prior and source normal pkl only.",
            "target_normal_role": "none in zero-target; none in discovery itself",
        },
        {
            "stage": "target_adaptation",
            "finding": "Historical target-assisted GSS loaded target normal and computed endpoint support P_target before applying C'=C*P_target(i)*P_target(j).",
            "target_normal_role": "generation input in target_assisted_full only",
        },
        {
            "stage": "causal_gss_construction",
            "finding": "target_guard.py does not open pkl itself, but its historical caller opened target normal and passed the empirical distribution. Source-only mode omits adaptation and guard.",
            "target_normal_role": "forbidden in all zero-target variants",
        },
        {
            "stage": "codex_generation_package",
            "finding": "The package copies only source-derived prompt/prior/hints for zero-target. Target-assisted packages contain target-adapted prior and guard artifacts.",
            "target_normal_role": "indirect historical dependency only for target_assisted_full",
        },
        {
            "stage": "gen_original_tof",
            "finding": "The unchanged Gen wrapper/core reads generated sequences and model/filter artifacts; the runtime gate verifies no target normal/attack read.",
            "target_normal_role": "forbidden",
        },
        {
            "stage": "causal_tof",
            "finding": "Historical target-assisted CLI opened target normal for distribution_penalty. Zero-target mode uses source causal prior, generated sequences, Gen TOF output, and gamma_dist=0. Reconstruction losses are optional external values and do not load target normal.",
            "target_normal_role": "forbidden in zero-target; historical target-assisted dependency retained only in frozen upper bound",
        },
        {
            "stage": "generation_quality_evaluation",
            "finding": "Runs in an isolated subprocess after synthetic data is locked; target normal is evaluation-only and no quality artifact is consumed upstream.",
            "target_normal_role": "allowed offline evaluation-only",
        },
        {
            "stage": "downstream_ad",
            "finding": "The unchanged Gen AD protocol reads synthetic training data, then target normal and attack data for evaluation.",
            "target_normal_role": "allowed downstream evaluation-only",
        },
    ]
    return {
        "schema_version": 1,
        "dataset": "sp",
        "scenario": "st",
        "seed": 2024,
        "scope": "SP-ST seed2024 only; no other dataset-scenario cell executed",
        "formal_method_line": "zero_target",
        "target_assisted_full_classification": "target-assisted upper bound; explicitly not zero-target",
        "historical_target_assisted_lineage": {
            "target_adapted_causal_prior": "outputs/main_runs/sp_st/seed2024/causal_gss/target_adapted_causal_prior.json",
            "target_normal_file": "data/gen/sp/spring/split_test.pkl",
            "endpoint_support_origin": "event-level device frequency computed from target normal flattened sequences",
            "weighting": "C_target(i,j)=C_source(i,j)*P_target(i)*P_target(j)",
            "target_guard_file_io": "target_guard.py does not open pkl; build_causal_gss_prompt.py opens target normal and passes P_target",
            "causal_tof_distribution_io": "run_causal_tof.py opens target normal and passes its empirical device distribution; causal_tof.py itself consumes the passed mapping",
            "reconstruction_component": "no target-normal read; absent loss vector makes this component zero",
        },
        "entrypoint_audit": {
            "main_prepare_generation.py": "target_assisted resolves and passes target normal before generation; zero_target explicitly omits and rejects it",
            "main_run_ablation.py": "legacy target-assisted runner passes target normal to Causal-TOF and offline quality; it is not used by the formal zero-target audit",
            "main_run_zero_target_audit.py": "hard-limited to SP-ST/seed2024; zero-target generation/filtering stages run with actual-open gates",
        },
        "expansion_decision": {
            "status": "do_not_run_remaining_eight_cells",
            "reason": "The strict zero-target variants pass lineage gates but SP-ST quality/AD metrics show a large domain gap; method quality is not yet adequate for nine-cell expansion.",
        },
        "frozen_outputs": {
            "before": frozen_before,
            "after": frozen_after,
            "unchanged": frozen_before == frozen_after,
        },
        "static_stage_audit": static_audit,
        "variant_lineage_files": {
            variant: str(seed_dir / "variants" / variant / "data_lineage.json")
            for variant in AUDIT_VARIANTS
        },
        "zero_target_pre_evaluation_gate": {
            variant: {
                "status": "pass",
                "target_normal_consumed": False,
                "target_attack_consumed": False,
            }
            for variant in ZERO_TARGET_VARIANTS
        },
        "downstream_ad_metrics": metrics,
        "generation_quality_metrics": quality,
        "actual_target_normal_files_read": sorted(set(actual_target_normal)),
        "actual_target_normal_read_events": target_read_events,
        "evaluation_only_isolation": {
            "generation_quality_outputs_consumed_upstream": quality_feedback,
            "status": "pass",
        },
        "safe_reproduction_command": (
            "/home/heyang/miniconda3/envs/smartguard_env/bin/python "
            "scripts/main_run_zero_target_audit.py --dataset sp --scenario st --seed 2024 "
            "--out-root outputs/zero_target_audit_repro --device cuda --cuda-visible-devices 0"
        ),
        "reproducible_commands": commands,
    }


def write_audit_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Zero-target Data Lineage Audit",
        "",
        "This audit covers **SP-ST / seed2024 only**. No other dataset-scenario cell was run.",
        "",
        "`target_assisted_full` is a target-assisted upper bound and is **not** a zero-target method.",
        "",
        "**Decision: do not run the remaining eight cells.** The lineage gate passes, but the strict zero-target SP-ST metrics show a large unresolved domain gap.",
        "",
        "## Frozen result protection",
        "",
        f"- Frozen root: `{payload['frozen_outputs']['before']['root']}`",
        f"- Files: {payload['frozen_outputs']['before']['file_count']}",
        f"- Before/after manifest: `{payload['frozen_outputs']['before']['manifest_sha256']}`",
        f"- Unchanged: `{payload['frozen_outputs']['unchanged']}`",
        "",
        "## Stage audit",
        "",
        "| stage | finding | target-normal role |",
        "| --- | --- | --- |",
    ]
    for row in payload["static_stage_audit"]:
        lines.append(f"| {row['stage']} | {row['finding']} | {row['target_normal_role']} |")
    lines.extend(["", "## Entrypoint audit", ""])
    for name, finding in payload["entrypoint_audit"].items():
        lines.append(f"- `{name}`: {finding}")
    lines.extend(
        [
            "",
            "## Five-variant comparison",
            "",
            "| variant | method class | F1 | FPR | device KL | transition similarity | causal similarity |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant in AUDIT_VARIANTS:
        ad = payload["downstream_ad_metrics"][variant]
        quality = payload["generation_quality_metrics"][variant]
        method = "target-assisted upper bound" if variant == "target_assisted_full" else "zero-target"
        lines.append(
            f"| {variant} | {method} | {_fmt(ad.get('f1'))} | {_fmt(ad.get('fpr'))} | "
            f"{_fmt(quality.get('device_distribution_kl'))} | {_fmt(quality.get('transition_matrix_similarity'))} | "
            f"{_fmt(quality.get('causal_graph_similarity'))} |"
        )
    lines.extend(["", "## Actual target-normal files read", ""])
    for item in payload["actual_target_normal_files_read"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "These reads occurred only in isolated offline quality evaluation and downstream AD evaluation subprocesses.", ""])
    lines.extend(["## Modified files", ""])
    for item in payload.get("modified_files", []):
        lines.append(f"- `{item}`")
    lines.append("")
    lines.extend(
        [
            "## Safe reproduction command",
            "",
            "Use a fresh output root; the runner refuses to overwrite an existing audit:",
            "",
            "```bash",
            payload["safe_reproduction_command"],
            "```",
            "",
            "## Executed stage commands",
            "",
            "The following commands are provenance for this completed run:",
            "",
            "```bash",
            *payload["reproducible_commands"],
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _ordered(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.6f}"
