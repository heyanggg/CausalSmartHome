#!/usr/bin/env python
"""Run the strict five-variant zero-target gate for SP-ST/seed2024 only."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.experiment_paths import source_pkl_for, target_pkl_for
from causal_smart_home.zero_target_audit import (
    AUDIT_VARIANTS,
    ZERO_TARGET_VARIANTS,
    aggregate_variant_lineage,
    assert_zero_target_pre_evaluation,
    audit_payload,
    lineage_environment,
    snapshot_tree,
    write_audit_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="sp", choices=["sp"])
    parser.add_argument("--scenario", default="st", choices=["st"])
    parser.add_argument("--seed", default=2024, type=int, choices=[2024])
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "outputs" / "zero_target_audit")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--dry-run-command", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_dir = args.out_root.resolve() / "sp_st" / "seed2024"
    if seed_dir.exists() and not args.dry_run_command:
        raise FileExistsError(
            f"audit output already exists and will not be overwritten: {seed_dir}"
        )

    frozen_root = REPO_ROOT / "outputs" / "main_runs" / "sp_st" / "seed2024"
    frozen_before = snapshot_tree(frozen_root)
    source_pkl = source_pkl_for("sp", "st")
    target_pkl = target_pkl_for("sp", "st")
    source_prior = REPO_ROOT / "data" / "main_experiment" / "sp_st" / "seed2024" / "causal_gss" / "resolved_causal_relation_prior.json"
    source_generated = REPO_ROOT / "outputs" / "zero_target_runs_v2_distribution_driven" / "sp_st" / "canonical_generation" / "version_v1" / "generated_codex.pkl"
    frozen_target_assisted = frozen_root / "causal_tof" / "generated_gen_tof_causal_tof.pkl"
    frozen_target_config = frozen_root / "causal_gss" / "config.json"
    shared = seed_dir / "shared"
    lineage_dir = seed_dir / "lineage"
    variants_dir = seed_dir / "variants"
    source_gss = shared / "source_only_causal_gss"
    package = shared / "source_only_generation_package"
    baseline_locked = shared / "zero_target_baseline_generation" / "generated_locked.pkl"
    gss_locked = shared / "zero_target_causal_gss_generation" / "generated_locked.pkl"
    baseline_gen_tof = shared / "baseline_gen_original_tof" / "gen_tof.pkl"
    gss_gen_tof = shared / "causal_gss_gen_original_tof" / "gen_tof.pkl"

    commands: list[str] = []
    stage_paths: dict[str, Path] = {}

    def run(
        name: str,
        command: list[str],
        *,
        stage: str,
        variant: str,
        purpose: str,
        allow_target_normal: bool = False,
        allow_target_attack: bool = False,
    ) -> None:
        rendered = " ".join(shlex.quote(part) for part in command)
        commands.append(rendered)
        lineage_path = lineage_dir / name / "data_lineage.json"
        stage_paths[name] = lineage_path
        print(rendered, flush=True)
        if args.dry_run_command:
            return
        env = lineage_environment(
            repo_root=REPO_ROOT,
            out_path=lineage_path,
            stage=stage,
            variant=variant,
            dataset="sp",
            scenario="st",
            seed=2024,
            purpose=purpose,
            allow_target_normal=allow_target_normal,
            allow_target_attack=allow_target_attack,
        )
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)

    py = sys.executable
    run(
        "source_causal_discovery_gss",
        [
            py, "scripts/build_causal_gss_prompt.py",
            "--source-pkl", str(source_pkl),
            "--prior-json", str(source_prior),
            "--adaptation-mode", "source_only",
            "--device-dict", str(REPO_ROOT / "data" / "gen" / "dictionary.py"),
            "--seed", "2024",
            "--out-prompt", str(source_gss / "prompt.txt"),
            "--out-prior-json", str(source_gss / "resolved_causal_relation_prior.json"),
            "--out-reweighted-hints", str(source_gss / "causal_reweighted_gss_hints.json"),
            "--out-config", str(source_gss / "config.json"),
        ],
        stage="causal_relation_discovery_and_causal_gss_construction",
        variant="zero_target_causal_gss+zero_target_full",
        purpose="generation",
    )
    run(
        "source_codex_generation_package",
        [
            py, "scripts/build_codex_generation_package.py",
            "--causal-gss-dir", str(source_gss),
            "--out-dir", str(package),
            "--scenario", "sp_st",
            "--seed", "2024",
        ],
        stage="codex_generation_package",
        variant="zero_target_causal_gss+zero_target_full",
        purpose="generation",
    )
    run(
        "baseline_synthetic_finalization",
        [
            py, "scripts/finalize_zero_target_synthetic.py",
            "--input-pkl", str(source_generated),
            "--out-pkl", str(baseline_locked),
            "--variant", "zero_target_baseline",
            "--seed", "2024",
            "--out-report", str(baseline_locked.parent / "generation_report.json"),
        ],
        stage="generation_and_synthetic_data_finalization",
        variant="zero_target_baseline+zero_target_causal_tof",
        purpose="generation",
    )
    run(
        "causal_gss_synthetic_finalization",
        [
            py, "scripts/finalize_zero_target_synthetic.py",
            "--input-pkl", str(source_generated),
            "--out-pkl", str(gss_locked),
            "--variant", "zero_target_causal_gss",
            "--source-causal-hints-json", str(source_gss / "causal_reweighted_gss_hints.json"),
            "--seed", "2024",
            "--out-report", str(gss_locked.parent / "generation_report.json"),
        ],
        stage="causal_gss_generation_and_synthetic_data_finalization",
        variant="zero_target_causal_gss+zero_target_full",
        purpose="generation",
    )
    run(
        "target_assisted_frozen_import",
        [
            py, "scripts/import_frozen_target_assisted.py",
            "--input-pkl", str(frozen_target_assisted),
            "--out-pkl", str(variants_dir / "target_assisted_full" / "synthetic_final.pkl"),
            "--out-report", str(variants_dir / "target_assisted_full" / "frozen_import_report.json"),
            "--historical-config", str(frozen_target_config),
            "--seed", "2024",
        ],
        stage="frozen_target_assisted_upper_bound_import",
        variant="target_assisted_full",
        purpose="generation",
    )
    run(
        "baseline_gen_original_tof",
        [
            py, "scripts/run_gen_original_tof.py",
            "--generated-pkl", str(baseline_locked),
            "--dataset", "sp", "--scenario", "st", "--seed", "2024",
            "--out-dir", str(baseline_gen_tof.parent), "--out-pkl", str(baseline_gen_tof),
            "--cuda-visible-devices", args.cuda_visible_devices,
        ],
        stage="gen_original_tof",
        variant="zero_target_baseline+zero_target_causal_tof",
        purpose="filtering",
    )
    run(
        "causal_gss_gen_original_tof",
        [
            py, "scripts/run_gen_original_tof.py",
            "--generated-pkl", str(gss_locked),
            "--dataset", "sp", "--scenario", "st", "--seed", "2024",
            "--out-dir", str(gss_gen_tof.parent), "--out-pkl", str(gss_gen_tof),
            "--cuda-visible-devices", args.cuda_visible_devices,
        ],
        stage="gen_original_tof",
        variant="zero_target_causal_gss+zero_target_full",
        purpose="filtering",
    )

    causal_tof_inputs = {
        "zero_target_causal_tof": baseline_gen_tof,
        "zero_target_full": gss_gen_tof,
    }
    for variant, generated in causal_tof_inputs.items():
        out = variants_dir / variant
        run(
            f"{variant}_causal_tof",
            [
                py, "scripts/run_causal_tof.py",
                "--generated-pkl", str(generated),
                "--guarded-hints-json", str(source_gss / "causal_reweighted_gss_hints.json"),
                "--method-line", "zero_target",
                "--gamma-dist", "0",
                "--alpha-rec", "1", "--beta-inconsistency", "1",
                "--out-scores", str(out / "causal_tof_scores.json"),
                "--out-weights", str(out / "causal_tof_weights.json"),
                "--out-weighted-resampled-pkl", str(out / "synthetic_final.pkl"),
                "--out-resampling-config", str(out / "causal_tof_config.json"),
                "--seed", "2024",
            ],
            stage="causal_tof",
            variant=variant,
            purpose="filtering",
        )

    if args.dry_run_command:
        return

    # The two non-Causal-TOF variants are locked by the unchanged Gen TOF output.
    final_inputs = {
        "zero_target_baseline": baseline_gen_tof,
        "zero_target_causal_gss": gss_gen_tof,
        "zero_target_causal_tof": variants_dir / "zero_target_causal_tof" / "synthetic_final.pkl",
        "zero_target_full": variants_dir / "zero_target_full" / "synthetic_final.pkl",
        "target_assisted_full": variants_dir / "target_assisted_full" / "synthetic_final.pkl",
    }
    for variant, synthetic in final_inputs.items():
        quality_dir = variants_dir / variant / "generation_quality"
        run(
            f"{variant}_generation_quality",
            [
                py, "scripts/evaluate_generation_quality.py",
                "--target-pkl", str(target_pkl), "--synthetic-pkl", str(synthetic),
                "--out-dir", str(quality_dir), "--dataset", "sp", "--scenario", "st",
                "--seed", "2024", "--variant", variant,
            ],
            stage="offline_generation_quality_evaluation",
            variant=variant,
            purpose="offline_generation_quality_evaluation",
            allow_target_normal=True,
        )
    for variant, synthetic in final_inputs.items():
        out = variants_dir / variant / "downstream_ad"
        pre_tof = baseline_locked if variant in {"zero_target_baseline", "zero_target_causal_tof"} else gss_locked
        gen_tof = baseline_gen_tof if variant in {"zero_target_baseline", "zero_target_causal_tof"} else gss_gen_tof
        if variant == "target_assisted_full":
            gen_tof = REPO_ROOT / "data" / "main_experiment" / "sp_st" / "seed2024" / "gen_original_tof" / "gen_tof.pkl"
        ad_command = [
            py, "scripts/run_gen_downstream_ad.py",
            "--dataset", "sp", "--scenario", "st", "--variant", variant,
            "--generated-pkl", str(synthetic),
            "--gen-tof-pkl", str(gen_tof), "--seed", "2024", "--out-dir", str(out),
            "--epochs", str(args.epochs), "--device", args.device,
            "--cuda-visible-devices", args.cuda_visible_devices,
        ]
        if variant != "target_assisted_full":
            ad_command.extend(["--pre-tof-pkl", str(pre_tof)])
        run(
            f"{variant}_downstream_ad",
            ad_command,
            stage="downstream_ad_evaluation",
            variant=variant,
            purpose="downstream_ad_evaluation",
            allow_target_normal=True,
            allow_target_attack=True,
        )

    lineage_map = {
        "zero_target_baseline": [
            stage_paths["baseline_synthetic_finalization"], stage_paths["baseline_gen_original_tof"],
            stage_paths["zero_target_baseline_generation_quality"], stage_paths["zero_target_baseline_downstream_ad"],
        ],
        "zero_target_causal_gss": [
            stage_paths["source_causal_discovery_gss"], stage_paths["source_codex_generation_package"],
            stage_paths["causal_gss_synthetic_finalization"], stage_paths["causal_gss_gen_original_tof"],
            stage_paths["zero_target_causal_gss_generation_quality"], stage_paths["zero_target_causal_gss_downstream_ad"],
        ],
        "zero_target_causal_tof": [
            stage_paths["baseline_synthetic_finalization"], stage_paths["baseline_gen_original_tof"],
            stage_paths["zero_target_causal_tof_causal_tof"], stage_paths["zero_target_causal_tof_generation_quality"],
            stage_paths["zero_target_causal_tof_downstream_ad"],
        ],
        "zero_target_full": [
            stage_paths["source_causal_discovery_gss"], stage_paths["source_codex_generation_package"],
            stage_paths["causal_gss_synthetic_finalization"], stage_paths["causal_gss_gen_original_tof"],
            stage_paths["zero_target_full_causal_tof"], stage_paths["zero_target_full_generation_quality"],
            stage_paths["zero_target_full_downstream_ad"],
        ],
        "target_assisted_full": [
            stage_paths["target_assisted_frozen_import"], stage_paths["target_assisted_full_generation_quality"],
            stage_paths["target_assisted_full_downstream_ad"],
        ],
    }
    aggregated = {}
    for variant in AUDIT_VARIANTS:
        aggregated[variant] = aggregate_variant_lineage(
            variant=variant, dataset="sp", scenario="st", seed=2024,
            lineage_paths=lineage_map[variant],
            out_path=variants_dir / variant / "data_lineage.json",
            inherited_target_dependency=variant == "target_assisted_full",
        )
    for variant in ZERO_TARGET_VARIANTS:
        assert_zero_target_pre_evaluation(aggregated[variant])

    frozen_after = snapshot_tree(frozen_root)
    if frozen_before != frozen_after:
        raise RuntimeError("frozen outputs/main_runs/sp_st/seed2024 changed during audit")
    payload = audit_payload(
        seed_dir=seed_dir,
        frozen_before=frozen_before,
        frozen_after=frozen_after,
        commands=commands,
        variant_lineages=aggregated,
    )
    tracked = subprocess.check_output(
        ["git", "diff", "--name-only"], cwd=REPO_ROOT, text=True
    ).splitlines()
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=REPO_ROOT, text=True
    ).splitlines()
    payload["modified_files"] = sorted(set(tracked + untracked))
    if not payload["frozen_outputs"]["unchanged"]:
        raise RuntimeError("frozen output verification failed")
    (seed_dir / "zero_target_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_audit_markdown(seed_dir / "zero_target_audit.md", payload)
    (seed_dir / "reproduce.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "OUT_ROOT=${1:-outputs/zero_target_audit_repro}\n"
        f"exec {sys.executable} scripts/main_run_zero_target_audit.py "
        "--dataset sp --scenario st --seed 2024 "
        "--out-root \"$OUT_ROOT\" --device cuda --cuda-visible-devices 0\n",
        encoding="utf-8",
    )
    print(f"saved audit: {seed_dir / 'zero_target_audit.md'}")


if __name__ == "__main__":
    main()
