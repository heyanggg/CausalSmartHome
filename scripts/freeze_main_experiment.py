#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = [2024, 2025, 2026]
SCENARIO_KEY = "sp_st"
DEFAULT_TAG = "20260623"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the current SP-ST GPT-5.5 main experiment.")
    parser.add_argument("--date-tag", default=DEFAULT_TAG)
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "outputs" / "main_experiment_frozen")
    parser.add_argument("--overwrite", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frozen_dir = (args.out_root / f"sp_st_gpt55_proposed_3seed_{args.date_tag}").resolve()
    if frozen_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"frozen directory already exists: {frozen_dir}")
        shutil.rmtree(frozen_dir)
    frozen_dir.mkdir(parents=True)

    copy_main_experiment(frozen_dir)
    copy_code_snapshot(frozen_dir)
    write_manifest(frozen_dir)
    write_docs(frozen_dir)
    write_reproduce_script(frozen_dir)
    write_provenance(frozen_dir)
    write_checksums(frozen_dir)
    print(f"frozen experiment saved: {frozen_dir}")


def copy_main_experiment(frozen_dir: Path) -> None:
    src = REPO_ROOT / "outputs" / "main_experiment"
    mapping = {
        "gpt55_generation": "generated",
        "gen_original_tof": "gen_original_tof",
        "causal_tof": "causal_tof",
        "downstream_ad": "downstream_ad",
        "summary": "summary",
        "causal_gss": "causal_gss",
    }
    for source_name, target_name in mapping.items():
        copytree(src / source_name, frozen_dir / target_name)


def copy_code_snapshot(frozen_dir: Path) -> None:
    dst = frozen_dir / "code_snapshot"
    copytree(REPO_ROOT / "scripts", dst / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    copytree(
        REPO_ROOT / "causal_smart_home",
        dst / "causal_smart_home",
        ignore=shutil.ignore_patterns("__pycache__", "gen_core", "resources"),
    )
    copytree(REPO_ROOT / "tests", dst / "tests", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    for name in ["README.md", "README_SELF.md", "pyproject.toml", "requirements.txt"]:
        src = REPO_ROOT / name
        if src.exists():
            shutil.copy2(src, dst / name)


def write_manifest(frozen_dir: Path) -> None:
    manifest: dict[str, Any] = {
        "proposed_method": "proposed_causal_gss_gpt55_causal_tof",
        "main_innovation": "causal-relation-enhanced GSS for GPT-5.5 behavior generation",
        "causal_tof_role": "post-TOF causal consistency enhancement component",
        "main_pipeline": [
            "causal relation prior",
            "target-distribution constraint",
            "causal-reweighted GSS",
            "GPT-5.5 generation",
            "Gen original two-stage TOF",
            "Causal-TOF",
            "Gen built-in downstream AD",
        ],
        "ablation_variants": {
            "ablation_no_causal_tof": "removes the post-TOF Causal-TOF component",
        },
        "removed_variants": {
            "raw_no_gen_tof": "removed from the current main experiment",
        },
        "scenario_key": SCENARIO_KEY,
        "dataset": "sp",
        "scenario": "st",
        "seeds": SEEDS,
        "frozen_dir": str(frozen_dir),
    }
    (frozen_dir / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def write_docs(frozen_dir: Path) -> None:
    readme = """# Frozen SP-ST GPT-5.5 Main Experiment

This directory freezes the current three-seed SP-ST main experiment.

It contains GPT-5.5 generated behavior sequences, Gen original two-stage TOF
outputs, Causal-TOF outputs, Gen built-in downstream AD runs for the proposed
method and the w/o Causal-TOF ablation, summary tables, provenance, checksums,
and a code snapshot.
"""
    reproduce = """# Reproduce From Frozen

Run from the repository root:

```bash
bash outputs/main_experiment_frozen/sp_st_gpt55_proposed_3seed_20260623/run_reproduce_from_frozen.sh
```

The script uses frozen GPT-5.5 pkl files and frozen causal GSS hints. It reruns
Gen original two-stage TOF, the w/o Causal-TOF downstream AD run, Causal-TOF,
the proposed-method downstream AD run, and summary. It does not regenerate
GPT-5.5 JSONL content.
"""
    (frozen_dir / "README_FROZEN.md").write_text(readme, encoding="utf-8")
    (frozen_dir / "REPRODUCE.md").write_text(reproduce, encoding="utf-8")


def write_reproduce_script(frozen_dir: Path) -> None:
    script = """#!/usr/bin/env bash
set -euo pipefail

FROZEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$FROZEN_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="sp"
SCENARIO="st"
SCENARIO_KEY="sp_st"
GEN_ROOT="$REPO_ROOT/causal_smart_home/gen_core"
TARGET_PKL="$REPO_ROOT/causal_smart_home/resources/gen_data/sp/spring/split_test.pkl"
OUT_ROOT="$FROZEN_DIR/reproduced_runs"

mkdir -p "$OUT_ROOT"
cd "$REPO_ROOT"

for SEED in 2024 2025 2026; do
  GEN_PKL="$FROZEN_DIR/generated/$SCENARIO_KEY/seed$SEED/generated_gpt55_clean.pkl"
  HINTS_JSON="$FROZEN_DIR/causal_gss/$SCENARIO_KEY/seed$SEED/guarded_reweighted_gss_hints.json"
  TOF_DIR="$OUT_ROOT/gen_original_tof/$SCENARIO_KEY/seed$SEED"
  CAUSAL_DIR="$OUT_ROOT/causal_tof/$SCENARIO_KEY/seed$SEED"
  AD_ROOT="$OUT_ROOT/downstream_ad/$SCENARIO_KEY/seed$SEED"

  PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" scripts/run_gen_original_tof.py \
    --generated-pkl "$GEN_PKL" \
    --dataset "$DATASET" \
    --scenario "$SCENARIO" \
    --seed "$SEED" \
    --gen-root "$GEN_ROOT" \
    --out-dir "$TOF_DIR" \
    --cuda-visible-devices 0

  PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" scripts/run_gen_downstream_ad.py \
    --dataset "$DATASET" \
    --scenario "$SCENARIO" \
    --variant ablation_no_causal_tof \
    --generated-pkl "$TOF_DIR/smartgen_tof.pkl" \
    --raw-generated-pkl "$GEN_PKL" \
    --seed "$SEED" \
    --out-dir "$AD_ROOT/ablation_no_causal_tof" \
    --gen-root "$GEN_ROOT" \
    --epochs 15 \
    --device cuda \
    --cuda-visible-devices 0

  PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" scripts/run_causal_tof.py \
    --generated-pkl "$TOF_DIR/smartgen_tof.pkl" \
    --guarded-hints-json "$HINTS_JSON" \
    --target-pkl "$TARGET_PKL" \
    --out-scores "$CAUSAL_DIR/causal_tof_scores.json" \
    --out-weights "$CAUSAL_DIR/generated.weights.json" \
    --out-weighted-resampled-pkl "$CAUSAL_DIR/generated_smartgen_tof_causal_tof.pkl" \
    --input-stage gen_original_tof \
    --mode weight \
    --temperature 2.0 \
    --seed "$SEED"

  PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" scripts/run_gen_downstream_ad.py \
    --dataset "$DATASET" \
    --scenario "$SCENARIO" \
    --variant proposed_causal_gss_gpt55_causal_tof \
    --generated-pkl "$CAUSAL_DIR/generated_smartgen_tof_causal_tof.pkl" \
    --raw-generated-pkl "$GEN_PKL" \
    --smartgen-tof-pkl "$TOF_DIR/smartgen_tof.pkl" \
    --seed "$SEED" \
    --out-dir "$AD_ROOT/proposed_causal_gss_gpt55_causal_tof" \
    --gen-root "$GEN_ROOT" \
    --epochs 15 \
    --device cuda \
    --cuda-visible-devices 0
done

PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" scripts/summarize_main_experiment.py \
  --runs-root "$OUT_ROOT/downstream_ad" \
  --out-dir "$OUT_ROOT/summary"
"""
    path = frozen_dir / "run_reproduce_from_frozen.sh"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def write_provenance(frozen_dir: Path) -> None:
    prov = frozen_dir / "provenance"
    prov.mkdir(parents=True, exist_ok=True)
    (prov / "project_status.txt").write_text("SP-ST GPT-5.5 main experiment frozen for seeds 2024, 2025, 2026.\n", encoding="utf-8")
    (prov / "git_status.txt").write_text(run_text(["git", "status", "--short"]), encoding="utf-8")
    (prov / "python_env.txt").write_text(run_text([sys.executable, "-c", "import sys, platform; print(sys.executable); print(sys.version); print(platform.platform())"]), encoding="utf-8")


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
