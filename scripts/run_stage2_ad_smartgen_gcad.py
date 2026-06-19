#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
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
CSH_ROOT = Path("/home/heyang/projects/CausalSmartHome")

DATASETS = {
    "fr_st": {
        "dataset": "fr",
        "env": "spring",
        "baseline_tof": SMARTGEN_ROOT / "SmartGen/filter_data/fr/spring/fr_spring_generation_SPPC_th=0.918_gpt-4o_seq_filter_true.pkl",
        "gcad_soft": CSH_ROOT / "outputs/gen_gcad/fr_st_device_soft/smartgen_tof_gcad_resampled.pkl",
        "attack": SMARTGEN_ROOT / "anomaly_detection_pipeline/attack/fr/labeled_fr_spring_attack_heater.pkl",
        "target_test": SMARTGEN_ROOT / "anomaly_detection_pipeline/test/fr/spring/test.pkl",
    },
    "sp_st": {
        "dataset": "sp",
        "env": "spring",
        "baseline_tof": SMARTGEN_ROOT / "SmartGen/filter_data/sp/spring/sp_spring_generation_SPPC_th=0.915_gpt-4o_seq_filter_true.pkl",
        "gcad_soft": CSH_ROOT / "outputs/gen_gcad/sp_st_device_soft/smartgen_tof_gcad_resampled.pkl",
        "attack": SMARTGEN_ROOT / "anomaly_detection_pipeline/attack/sp/labeled_sp_spring_attack_heater.pkl",
        "target_test": SMARTGEN_ROOT / "anomaly_detection_pipeline/test/sp/spring/test.pkl",
    },
}


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
        "result_path": payload.get("result_path"),
    }


def build_markdown(dataset_key: str, metrics: dict[str, Any]) -> str:
    lines = [
        f"# Stage 2 AD Metrics: {dataset_key}",
        "",
        "## Protocol",
        "",
        "- SmartGen TransformerAutoencoder AD pipeline via CausalSmartHome wrapper.",
        "- Synthetic pkl is split into train/validation with the same seed and split ratio for both comparison arms.",
        "- Threshold is the SmartGen validation-loss percentile rule.",
        "- Test set is target real normal plus labeled spring attack.",
        "- No SmartGuard training was run.",
        "",
        "## Metrics",
        "",
        "| group | precision | recall | F1 | FPR | FNR | accuracy | TP | TN | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics["rows"]:
        lines.append(
            f"| {row['name']} | {fmt(row['precision'])} | {fmt(row['recall'])} | {fmt(row['F1'])} | "
            f"{fmt(row['FPR'])} | {fmt(row['FNR'])} | {fmt(row['accuracy'])} | "
            f"{row['TP']} | {row['TN']} | {row['FP']} | {row['FN']} |"
        )
    lines.extend(["", "## Used Data", ""])
    for key, value in metrics["used_data_paths"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Settings", ""])
    for key, value in metrics["settings"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def run_one_group(
    dataset_key: str,
    group: str,
    synthetic_pkl: Path,
    info: dict[str, Any],
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    tag = f"{dataset_key}_{group}"
    config = SmartGenAnomalyRunConfig(
        smartgen_root=SMARTGEN_ROOT,
        dataset=info["dataset"],
        env=info["env"],
        synthetic_pkl=synthetic_pkl.resolve(),
        out_dir=out_dir,
        tag=tag,
        epochs=args.epochs,
        seed=args.seed,
        split_ratio=args.split_ratio,
        device=args.device,
        cuda_visible_devices=args.cuda_visible_devices,
        dry_run=args.dry_run,
        attack_pkl=info["attack"].resolve(),
        target_test_pkl=info["target_test"].resolve(),
        threshold_percentage=args.threshold_percentage,
    )
    log_path = out_dir / f"{tag}_train_command.log"
    with open(log_path, "w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        print("Stage 2 SmartGen AD run")
        print(json.dumps(jsonable(config.__dict__), ensure_ascii=False, indent=2))
        payload = run_smartgen_anomaly_experiment(config)
        print("Payload")
        print(json.dumps(jsonable(payload), ensure_ascii=False, indent=2))
    payload = add_rates(payload)
    payload["train_command_log"] = str(log_path)
    result_copy = out_dir / f"{tag}_metrics_payload.json"
    result_copy.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def run_dataset(dataset_key: str, args: argparse.Namespace) -> dict[str, Any]:
    info = DATASETS[dataset_key]
    out_dir = CSH_ROOT / "outputs/stage2_ad" / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    for required in ("baseline_tof", "gcad_soft", "attack", "target_test"):
        if not info[required].exists():
            raise FileNotFoundError(f"{dataset_key} {required} not found: {info[required]}")

    threshold_percentage = (
        args.threshold_percentage
        if args.threshold_percentage is not None
        else DEFAULT_THRESHOLD_PERCENTAGES[(info["dataset"], info["env"])]
    )
    args_for_run = argparse.Namespace(**vars(args))
    args_for_run.threshold_percentage = threshold_percentage

    groups = {
        "smartgen_tof": info["baseline_tof"],
        "device_gcad_soft": info["gcad_soft"],
    }
    payloads = {
        name: run_one_group(dataset_key, name, path, info, out_dir, args_for_run)
        for name, path in groups.items()
    }
    metrics = {
        "dataset_key": dataset_key,
        "settings": {
            "epochs": args.epochs,
            "seed": args.seed,
            "split_ratio": args.split_ratio,
            "device": args.device,
            "cuda_visible_devices": args.cuda_visible_devices,
            "threshold_percentage": threshold_percentage,
            "dry_run": args.dry_run,
        },
        "used_data_paths": {key: str(value.resolve()) for key, value in info.items() if isinstance(value, Path)},
        "rows": [metric_row(name, payload) for name, payload in payloads.items()],
        "payloads": payloads,
        "auroc_auprc": "not supported by the reused SmartGen TransformerAutoencoder evaluation wrapper",
    }
    (out_dir / "metrics.json").write_text(json.dumps(jsonable(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metrics.md").write_text(build_markdown(dataset_key, metrics), encoding="utf-8")
    (out_dir / "used_data_paths.json").write_text(json.dumps(jsonable(metrics["used_data_paths"]), ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run Stage 2 SmartGen AD comparison for TOF vs device-GCAD soft")
    parser.add_argument("--datasets", default="fr_st,sp_st", help="comma-separated subset: fr_st,sp_st")
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
    selected = [item.strip() for item in args.datasets.split(",") if item.strip()]
    unknown = [item for item in selected if item not in DATASETS]
    if unknown:
        raise ValueError(f"unknown dataset key(s): {unknown}")
    all_metrics = {dataset_key: run_dataset(dataset_key, args) for dataset_key in selected}
    summary_path = CSH_ROOT / "outputs/stage2_ad/stage2_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(jsonable(all_metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "datasets": selected, "dry_run": args.dry_run}, indent=2))


if __name__ == "__main__":
    main()
