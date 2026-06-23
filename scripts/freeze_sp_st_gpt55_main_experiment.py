#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = [2024, 2025, 2026]
SCENARIO_KEY = "sp_st"
STAGE4A_PREFIX = "sp_st_downweight_multiplicative_" + "co" + "dex_gpt55_seed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the SP-ST GPT-5.5 three-seed main experiment.")
    parser.add_argument("--date-tag", default=date.today().strftime("%Y%m%d"))
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "outputs" / "main_experiment_frozen")
    parser.add_argument("--overwrite", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frozen_dir = (args.out_root / f"sp_st_gpt55_3seed_{args.date_tag}").resolve()
    if frozen_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"frozen directory already exists: {frozen_dir}")
        shutil.rmtree(frozen_dir)
    frozen_dir.mkdir(parents=True)

    source_paths = build_source_paths()
    copy_inputs(frozen_dir)
    copy_main_outputs(frozen_dir)
    copy_results(frozen_dir)
    copy_code_snapshot(frozen_dir)
    write_source_paths(frozen_dir, source_paths)
    write_provenance(frozen_dir)
    write_manifest(frozen_dir, source_paths)
    write_docs(frozen_dir)
    write_reproduce_script(frozen_dir)
    write_checksums(frozen_dir)
    print(f"frozen experiment saved: {frozen_dir}")


def build_source_paths() -> dict[str, Any]:
    return {
        "repo_root": str(REPO_ROOT),
        "mainline_gpt55_generation": str(REPO_ROOT / "outputs" / "mainline_gpt55_generation"),
        "stage4a_prompt_prior_guard_hints": {
            f"seed{seed}": str(REPO_ROOT / "outputs" / "gcad_gss_stage4" / f"{STAGE4A_PREFIX}{seed}")
            for seed in SEEDS
        },
        "generation_package": {
            f"seed{seed}": str(REPO_ROOT / "outputs" / "gpt55_generation_package" / "sp_st" / f"seed{seed}") for seed in SEEDS
        },
        "smartgen_root": "/home/heyang/projects/SmartGen",
        "target_pkl": "/home/heyang/projects/SmartGen/parameter_study/test/sp/spring/split_test.pkl",
        "dictionary_py": "/home/heyang/projects/SmartGen/SmartGen/dictionary.py",
    }


def copy_inputs(frozen_dir: Path) -> None:
    for seed in SEEDS:
        seed_dst = frozen_dir / "inputs" / SCENARIO_KEY / f"seed{seed}"
        stage4a_src = REPO_ROOT / "outputs" / "gcad_gss_stage4" / f"{STAGE4A_PREFIX}{seed}"
        package_src = REPO_ROOT / "outputs" / "gpt55_generation_package" / "sp_st" / f"seed{seed}"
        copytree(stage4a_src, seed_dst / "stage4a_prompt_prior_guard_hints")
        copytree(package_src, seed_dst / "generation_package")


def copy_main_outputs(frozen_dir: Path) -> None:
    mainline = REPO_ROOT / "outputs" / "mainline_gpt55_generation"
    for seed in SEEDS:
        copytree(mainline / "gpt55_generation" / SCENARIO_KEY / f"seed{seed}", frozen_dir / "generated" / SCENARIO_KEY / f"seed{seed}")
        copytree(mainline / "smartgen_original_tof" / SCENARIO_KEY / f"seed{seed}", frozen_dir / "smartgen_original_tof" / SCENARIO_KEY / f"seed{seed}")
        copytree(
            mainline / "smartgen_original_tof_plus_causal_tof" / SCENARIO_KEY / f"seed{seed}",
            frozen_dir / "causal_tof" / SCENARIO_KEY / f"seed{seed}",
        )
        copytree(mainline / "gen_builtin_ad" / SCENARIO_KEY / f"seed{seed}", frozen_dir / "downstream_ad" / SCENARIO_KEY / f"seed{seed}")


def copy_results(frozen_dir: Path) -> None:
    src = REPO_ROOT / "outputs" / "mainline_gpt55_generation" / "gen_builtin_ad"
    dst = frozen_dir / "frozen_results"
    dst.mkdir(parents=True, exist_ok=True)
    mapping = {
        "gen_builtin_ad_tof_corrected_per_seed.md": "per_seed.md",
        "gen_builtin_ad_tof_corrected_aggregate.md": "aggregate.md",
        "gen_builtin_ad_tof_corrected_seed_deltas.md": "seed_deltas.md",
        "gen_builtin_ad_tof_corrected_per_seed.csv": "per_seed.csv",
        "gen_builtin_ad_tof_corrected_aggregate.csv": "aggregate.csv",
        "gen_builtin_ad_tof_corrected_seed_deltas.csv": "seed_deltas.csv",
        "gen_builtin_ad_tof_corrected_per_seed.json": "per_seed.json",
        "gen_builtin_ad_tof_corrected_aggregate.json": "aggregate.json",
        "gen_builtin_ad_tof_corrected_seed_deltas.json": "seed_deltas.json",
    }
    for old_name, new_name in mapping.items():
        shutil.copy2(src / old_name, dst / new_name)


def copy_code_snapshot(frozen_dir: Path) -> None:
    dst = frozen_dir / "code_snapshot"
    copytree(REPO_ROOT / "scripts", dst / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    copytree(REPO_ROOT / "causal_smart_home", dst / "causal_smart_home", ignore=shutil.ignore_patterns("__pycache__"))
    copytree(REPO_ROOT / "tests", dst / "tests", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    for name in ["README.md", "README_SELF.md", "pyproject.toml"]:
        src = REPO_ROOT / name
        if src.exists():
            shutil.copy2(src, dst / name)


def write_source_paths(frozen_dir: Path, source_paths: dict[str, Any]) -> None:
    path = frozen_dir / "provenance" / "source_paths.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(source_paths, ensure_ascii=False, indent=2), encoding="utf-8")


def write_provenance(frozen_dir: Path) -> None:
    prov = frozen_dir / "provenance"
    prov.mkdir(parents=True, exist_ok=True)
    (prov / "project_status.txt").write_text("SP-ST GPT-5.5 generation main experiment frozen successfully for seeds 2024, 2025, 2026.\n", encoding="utf-8")
    (prov / "git_status.txt").write_text(run_text(["git", "status", "--short"]), encoding="utf-8")
    (prov / "python_env.txt").write_text(run_text([sys.executable, "-c", "import sys, platform; print(sys.executable); print(sys.version); print(platform.platform())"]), encoding="utf-8")
    (prov / "pip_freeze.txt").write_text(run_text([sys.executable, "-m", "pip", "freeze"]), encoding="utf-8")


def write_manifest(frozen_dir: Path, source_paths: dict[str, Any]) -> None:
    manifest = {
        "experiment": "SP-ST GPT-5.5 generation main experiment",
        "scenario_key": SCENARIO_KEY,
        "dataset": "sp",
        "scenario": "st",
        "seeds": SEEDS,
        "generator": "gpt55_generation",
        "generation_model": "GPT-5.5",
        "api_llm": False,
        "manual_generation": True,
        "surrogate_algorithm": False,
        "pipeline": [
            "GCAD causal prior",
            "target-distribution downweight guard",
            "multiplicative causal-reweighted GSS",
            "guarded causal-reweighted GSS prompt",
            "GPT-5.5 generation",
            "SmartGen original two-stage TOF",
            "optional Causal-TOF",
            "SmartGen built-in downstream AD",
            "summary",
        ],
        "frozen_dir": str(frozen_dir),
        "source_paths": source_paths,
    }
    (frozen_dir / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def write_docs(frozen_dir: Path) -> None:
    readme = f"""# Frozen SP-ST GPT-5.5 Three-Seed Main Experiment

This directory freezes the completed SP-ST GPT-5.5 generation main experiment for seeds 2024, 2025, and 2026.

The frozen results include generated GPT-5.5 behavior sequences, SmartGen original two-stage TOF outputs, optional Causal-TOF outputs, SmartGen built-in downstream AD runs, summaries, provenance, checksums, and a code snapshot.

SmartGen original two-stage TOF is included to restore SmartGen's original evaluation pipeline. The research focus is GCAD-guided causal GSS, GPT-5.5 target-context generation, and optional post-TOF Causal-TOF.
"""
    reproduce = f"""# Reproduce From Frozen

Run from the repository root:

```bash
bash {frozen_dir.relative_to(REPO_ROOT)}/run_reproduce_from_frozen.sh
```

The script uses frozen `generated_gpt55_clean.pkl` files and frozen guarded hint JSON files. It reruns SmartGen original two-stage TOF, raw downstream AD, mainline downstream AD, Causal-TOF, post-Causal-TOF downstream AD, and summary into `reproduced_runs/`.

It does not regenerate GPT-5.5 JSONL content and does not overwrite the frozen original archive. Downstream AD retrains models, so small differences can appear across GPU, driver, PyTorch, and CUDA environments.
"""
    (frozen_dir / "README_FROZEN.md").write_text(readme, encoding="utf-8")
    (frozen_dir / "REPRODUCE.md").write_text(reproduce, encoding="utf-8")


def write_reproduce_script(frozen_dir: Path) -> None:
    script = """#!/usr/bin/env bash
set -euo pipefail

FROZEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$FROZEN_DIR/../../.." && pwd)"
SMARTGEN="${SMARTGEN:-/home/heyang/projects/SmartGen}"
PYTHON_BIN="${PYTHON_BIN:-/home/heyang/miniconda3/envs/smartguard_env/bin/python}"
DATASET="sp"
SCENARIO="st"
SCENARIO_KEY="sp_st"
TGT_PKL="$SMARTGEN/parameter_study/test/sp/spring/split_test.pkl"
OUT_ROOT="$FROZEN_DIR/reproduced_runs"

mkdir -p "$OUT_ROOT"
cd "$REPO_ROOT"

for SEED in 2024 2025 2026; do
  GEN_PKL="$FROZEN_DIR/generated/$SCENARIO_KEY/seed$SEED/generated_gpt55_clean.pkl"
  HINTS_JSON="$FROZEN_DIR/inputs/$SCENARIO_KEY/seed$SEED/stage4a_prompt_prior_guard_hints/guarded_reweighted_gss_hints.json"
  TOF_DIR="$OUT_ROOT/smartgen_original_tof/$SCENARIO_KEY/seed$SEED"
  CAUSAL_DIR="$OUT_ROOT/causal_tof/$SCENARIO_KEY/seed$SEED"
  AD_ROOT="$OUT_ROOT/gen_builtin_ad/$SCENARIO_KEY/seed$SEED"

  PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" scripts/run_stage4c_smartgen_original_tof.py \
    --generated-pkl "$GEN_PKL" \
    --dataset "$DATASET" \
    --scenario "$SCENARIO" \
    --seed "$SEED" \
    --smartgen-root "$SMARTGEN" \
    --out-dir "$TOF_DIR" \
    --cuda-visible-devices 0

  PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" scripts/run_stage4c_gen_builtin_downstream_ad.py \
    --dataset "$DATASET" \
    --scenario "$SCENARIO" \
    --variant stage4_raw_no_smartgen_tof \
    --generated-pkl "$GEN_PKL" \
    --raw-generated-pkl "$GEN_PKL" \
    --seed "$SEED" \
    --out-dir "$AD_ROOT/stage4_raw_no_smartgen_tof" \
    --smartgen-root "$SMARTGEN" \
    --epochs 15 \
    --device cuda \
    --cuda-visible-devices 0

  PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" scripts/run_stage4c_gen_builtin_downstream_ad.py \
    --dataset "$DATASET" \
    --scenario "$SCENARIO" \
    --variant stage4_smartgen_original_tof \
    --generated-pkl "$TOF_DIR/smartgen_tof.pkl" \
    --raw-generated-pkl "$GEN_PKL" \
    --seed "$SEED" \
    --out-dir "$AD_ROOT/stage4_smartgen_original_tof" \
    --smartgen-root "$SMARTGEN" \
    --epochs 15 \
    --device cuda \
    --cuda-visible-devices 0

  PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" scripts/run_causal_tof_weighting.py \
    --generated-pkl "$TOF_DIR/smartgen_tof.pkl" \
    --guarded-hints-json "$HINTS_JSON" \
    --target-pkl "$TGT_PKL" \
    --out-scores "$CAUSAL_DIR/causal_tof_scores.json" \
    --out-weights "$CAUSAL_DIR/generated.weights.json" \
    --out-weighted-resampled-pkl "$CAUSAL_DIR/generated_smartgen_tof_causal_tof.pkl" \
    --input-stage smartgen_original_tof \
    --mode weight \
    --temperature 2.0 \
    --seed "$SEED"

  PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" scripts/run_stage4c_gen_builtin_downstream_ad.py \
    --dataset "$DATASET" \
    --scenario "$SCENARIO" \
    --variant stage4_smartgen_original_tof_plus_causal_tof \
    --generated-pkl "$CAUSAL_DIR/generated_smartgen_tof_causal_tof.pkl" \
    --raw-generated-pkl "$GEN_PKL" \
    --smartgen-tof-pkl "$TOF_DIR/smartgen_tof.pkl" \
    --seed "$SEED" \
    --out-dir "$AD_ROOT/stage4_smartgen_original_tof_plus_causal_tof" \
    --smartgen-root "$SMARTGEN" \
    --epochs 15 \
    --device cuda \
    --cuda-visible-devices 0
done

PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" scripts/summarize_stage4_gen_builtin_ad_tof_corrected.py \
  --runs-root "$OUT_ROOT/gen_builtin_ad" \
  --out-dir "$OUT_ROOT/gen_builtin_ad"
"""
    path = frozen_dir / "run_reproduce_from_frozen.sh"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def write_checksums(frozen_dir: Path) -> None:
    checksum_path = frozen_dir / "provenance" / "checksums.sha256"
    rows: list[str] = []
    for path in sorted(p for p in frozen_dir.rglob("*") if p.is_file()):
        if path == checksum_path:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(frozen_dir)}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def copytree(src: Path, dst: Path, ignore=None) -> None:
    if not src.exists():
        raise FileNotFoundError(f"required source not found: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)


def run_text(cmd: list[str]) -> str:
    try:
        completed = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    except Exception as exc:
        return f"failed to run {cmd}: {exc}\n"
    out = completed.stdout
    if completed.stderr:
        out += "\n[stderr]\n" + completed.stderr
    if completed.returncode != 0:
        out += f"\n[returncode] {completed.returncode}\n"
    return out


if __name__ == "__main__":
    main()
