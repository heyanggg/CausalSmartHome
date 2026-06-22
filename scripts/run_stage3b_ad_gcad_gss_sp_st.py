#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.smartgen_experiment import (
    DEFAULT_THRESHOLD_PERCENTAGES,
    SmartGenAnomalyRunConfig,
    run_smartgen_anomaly_experiment,
)


SMARTGEN_ROOT = Path("/home/heyang/projects/SmartGen")
CSH_ROOT = REPO_ROOT
DEFAULT_STAGE3A_TAG = "sp_st_codex_calibrated_seed2024"
DATASET = "sp"
ENV = "spring"


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def add_rates(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    fp = int(out.get("FP", 0) or 0)
    tn = int(out.get("TN", 0) or 0)
    fn = int(out.get("FN", 0) or 0)
    tp = int(out.get("TP", 0) or 0)
    out["FPR"] = fp / (fp + tn) if fp + tn else 0.0
    out["FNR"] = fn / (fn + tp) if fn + tp else 0.0
    out["F1"] = out.get("F1 score", out.get("F1", 0.0))
    return out


def metric_row(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    payload = add_rates(payload)
    return {
        "name": name,
        "precision": payload.get("precision"),
        "recall": payload.get("recall"),
        "F1": payload.get("F1"),
        "FPR": payload.get("FPR"),
        "FNR": payload.get("FNR"),
        "accuracy": payload.get("accuracy"),
        "TP": payload.get("TP"),
        "TN": payload.get("TN"),
        "FP": payload.get("FP"),
        "FN": payload.get("FN"),
        "learned_threshold": payload.get("learned_threshold"),
        "synthetic_size": payload.get("synthetic_size"),
        "train_size": payload.get("train_size"),
        "vld_size": payload.get("vld_size"),
        "threshold_vld_size": payload.get("threshold_vld_size"),
        "test_size": payload.get("test_size"),
        "normal_test_size": payload.get("normal_test_size"),
        "attack_test_size": payload.get("attack_test_size"),
        "result_path": payload.get("result_path"),
        "train_command_log": payload.get("train_command_log"),
    }


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def delta_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {row["name"]: row for row in rows}
    original = by_name.get("original_prompt")
    enhanced = by_name.get("enhanced_prompt")
    if not original or not enhanced:
        return {}
    metrics = ["precision", "recall", "F1", "FPR", "FNR", "accuracy", "learned_threshold"]
    return {
        f"delta_{metric}": (
            enhanced.get(metric) - original.get(metric)
            if isinstance(enhanced.get(metric), (int, float)) and isinstance(original.get(metric), (int, float))
            else None
        )
        for metric in metrics
    }


def build_markdown(metrics: dict[str, Any]) -> str:
    lines = [
        "# Stage 3B AD Check: SP-ST GCAD-GSS Prompt",
        "",
        "## Protocol",
        "",
        "- Dataset: SP-ST only.",
        "- Compared synthetic spring sequences from Stage 3A after SmartGen TOF.",
        "- The two arms reuse the same SmartGen AD wrapper, target test data, attack data, seed, split ratio, epochs, and threshold percentile.",
        "- No SmartGuard training was run.",
        "- This is a downstream AD check, not a claim that GCAD-GSS improves every AD setting.",
        "",
        "## Metrics",
        "",
        "| group | precision | recall | F1 | FPR | FNR | accuracy | TP | TN | FP | FN | threshold | synthetic |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics["rows"]:
        lines.append(
            f"| {row['name']} | {fmt(row['precision'])} | {fmt(row['recall'])} | {fmt(row['F1'])} | "
            f"{fmt(row['FPR'])} | {fmt(row['FNR'])} | {fmt(row['accuracy'])} | "
            f"{row['TP']} | {row['TN']} | {row['FP']} | {row['FN']} | "
            f"{fmt(row['learned_threshold'])} | {row['synthetic_size']} |"
        )
    lines.extend(["", "## Delta Enhanced Minus Original", ""])
    for key, value in metrics["delta_enhanced_minus_original"].items():
        lines.append(f"- {key}: `{fmt(value)}`")
    lines.extend(["", "## Used Data", ""])
    for key, value in metrics["used_data_paths"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Settings", ""])
    for key, value in metrics["settings"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "name",
        "precision",
        "recall",
        "F1",
        "FPR",
        "FNR",
        "accuracy",
        "TP",
        "TN",
        "FP",
        "FN",
        "learned_threshold",
        "synthetic_size",
        "train_size",
        "vld_size",
        "threshold_vld_size",
        "test_size",
        "normal_test_size",
        "attack_test_size",
        "result_path",
        "train_command_log",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_one_group(
    group: str,
    synthetic_pkl: Path,
    out_dir: Path,
    args: argparse.Namespace,
    threshold_percentage: float,
) -> dict[str, Any]:
    tag = f"sp_st_gcad_gss_{group}"
    config = SmartGenAnomalyRunConfig(
        smartgen_root=SMARTGEN_ROOT,
        dataset=DATASET,
        env=ENV,
        synthetic_pkl=synthetic_pkl.resolve(),
        out_dir=out_dir,
        tag=tag,
        epochs=args.epochs,
        seed=args.seed,
        split_ratio=args.split_ratio,
        device=args.device,
        cuda_visible_devices=args.cuda_visible_devices,
        dry_run=args.dry_run,
        attack_pkl=args.attack_pkl.resolve(),
        target_test_pkl=args.target_test_pkl.resolve(),
        threshold_percentage=threshold_percentage,
    )
    log_path = out_dir / f"{tag}_train_command.log"
    with open(log_path, "w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        print("Stage 3B SmartGen AD run")
        print(json.dumps(jsonable(config.__dict__), ensure_ascii=False, indent=2))
        payload = run_smartgen_anomaly_experiment(config)
        print("Payload")
        print(json.dumps(jsonable(payload), ensure_ascii=False, indent=2))
    payload = add_rates(payload)
    payload["train_command_log"] = str(log_path)
    payload_path = out_dir / f"{tag}_metrics_payload.json"
    payload_path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run Stage 3B SP-ST AD check for GCAD-GSS prompt outputs")
    parser.add_argument("--stage3a-tag", default=DEFAULT_STAGE3A_TAG)
    parser.add_argument("--stage3a-root", type=Path)
    parser.add_argument("--original-tof-pkl", type=Path)
    parser.add_argument("--enhanced-tof-pkl", type=Path)
    parser.add_argument("--out-dir", type=Path, default=CSH_ROOT / "outputs/gcad_gss/sp_st_stage3b_ad")
    parser.add_argument(
        "--attack-pkl",
        type=Path,
        default=SMARTGEN_ROOT / "anomaly_detection_pipeline/attack/sp/labeled_sp_spring_attack_heater.pkl",
    )
    parser.add_argument(
        "--target-test-pkl",
        type=Path,
        default=SMARTGEN_ROOT / "anomaly_detection_pipeline/test/sp/spring/test.pkl",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--threshold-percentage", type=float)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage3a_root = args.stage3a_root or (CSH_ROOT / "outputs/gcad_gss" / args.stage3a_tag)
    original_tof = args.original_tof_pkl or (stage3a_root / "sp_st_original/smartgen_tof.pkl")
    enhanced_tof = args.enhanced_tof_pkl or (stage3a_root / "sp_st_enhanced/smartgen_tof.pkl")
    out_dir = (args.out_dir / args.stage3a_tag).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    required_paths = {
        "original_tof": original_tof,
        "enhanced_tof": enhanced_tof,
        "attack_pkl": args.attack_pkl,
        "target_test_pkl": args.target_test_pkl,
    }
    missing = {key: str(path) for key, path in required_paths.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(f"missing required path(s): {missing}")

    threshold_percentage = (
        args.threshold_percentage
        if args.threshold_percentage is not None
        else DEFAULT_THRESHOLD_PERCENTAGES[(DATASET, ENV)]
    )

    payloads = {
        "original_prompt": run_one_group("original_prompt", original_tof, out_dir, args, threshold_percentage),
        "enhanced_prompt": run_one_group("enhanced_prompt", enhanced_tof, out_dir, args, threshold_percentage),
    }
    rows = [metric_row(name, payload) for name, payload in payloads.items()]
    metrics = {
        "dataset_key": "sp_st",
        "stage3a_tag": args.stage3a_tag,
        "settings": {
            "epochs": args.epochs,
            "seed": args.seed,
            "split_ratio": args.split_ratio,
            "device": args.device,
            "cuda_visible_devices": args.cuda_visible_devices,
            "threshold_percentage": threshold_percentage,
            "dry_run": args.dry_run,
        },
        "used_data_paths": {key: str(path.resolve()) for key, path in required_paths.items()},
        "rows": rows,
        "delta_enhanced_minus_original": delta_rows(rows),
        "payloads": payloads,
    }
    (out_dir / "metrics.json").write_text(json.dumps(jsonable(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metrics.md").write_text(build_markdown(metrics), encoding="utf-8")
    write_csv(out_dir / "metrics.csv", rows)
    print(json.dumps({"out_dir": str(out_dir), "dry_run": args.dry_run}, indent=2))


if __name__ == "__main__":
    main()
