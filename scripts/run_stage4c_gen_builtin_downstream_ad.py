#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.smartgen_experiment import (
    DEFAULT_THRESHOLD_PERCENTAGES,
    SmartGenAnomalyRunConfig,
    default_smartgen_paths,
    load_pickle,
    run_smartgen_anomaly_experiment,
)


VARIANTS = {
    "stage3_prompt_only_baseline",
    "stage4_downweight_multiplicative_raw",
    "stage4_downweight_multiplicative_causal_tof_resampled",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage4C using SmartGen built-in anomaly_detection_pipeline semantics.")
    parser.add_argument("--dataset", required=True, choices=["fr", "sp"])
    parser.add_argument("--scenario", required=True, choices=["st"], help="CausalSmartHome scenario suffix; st maps to SmartGen spring.")
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--generated-pkl", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--smartgen-root", type=Path, default=Path("/home/heyang/projects/SmartGen"))
    parser.add_argument("--causal-smart-home-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--threshold-percentage", type=float)
    parser.add_argument("--validation-pkl", type=Path)
    parser.add_argument("--attack-pkl", type=Path)
    parser.add_argument("--target-test-pkl", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def fpr(payload: dict[str, Any]) -> float | None:
    fp = payload.get("FP")
    tn = payload.get("TN")
    if fp is None or tn is None:
        return None
    denom = float(fp) + float(tn)
    return float(fp) / denom if denom else None


def fnr(payload: dict[str, Any]) -> float | None:
    fn = payload.get("FN")
    tp = payload.get("TP")
    if fn is None or tp is None:
        return None
    denom = float(fn) + float(tp)
    return float(fn) / denom if denom else None


def f1(payload: dict[str, Any]) -> float | None:
    value = payload.get("F1 score", payload.get("F1"))
    return float(value) if value is not None else None


def generator_for_variant(variant: str) -> str:
    if variant == "stage3_prompt_only_baseline":
        return "codex_calibrated_stage3"
    return "codex_gpt55_surrogate"


def normalize_metrics(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    threshold_percentage = (
        args.threshold_percentage
        if args.threshold_percentage is not None
        else DEFAULT_THRESHOLD_PERCENTAGES[(args.dataset, smartgen_env(args.scenario))]
    )
    return {
        "status": "success",
        "precision": payload.get("precision"),
        "recall": payload.get("recall"),
        "f1": f1(payload),
        "accuracy": payload.get("accuracy"),
        "fpr": fpr(payload),
        "fnr": fnr(payload),
        "threshold": payload.get("learned_threshold"),
        "threshold_source": f"validation_percentile_{threshold_percentage}",
        "train_size": payload.get("train_size"),
        "validation_size": payload.get("threshold_vld_size", payload.get("vld_size")),
        "test_size": payload.get("test_size"),
        "generated_size": payload.get("synthetic_size"),
        "seed": args.seed,
        "dataset": args.dataset,
        "scenario": args.scenario,
        "smartgen_env": smartgen_env(args.scenario),
        "variant": args.variant,
        "downstream_pipeline": "smartgen_builtin_anomaly_detection_pipeline",
        "generator": generator_for_variant(args.variant),
        "api_llm": False,
        "surrogate_for_smartgen_llm": args.variant != "stage3_prompt_only_baseline",
        "epochs": args.epochs,
        "split_ratio": args.split_ratio,
        "device": payload.get("device"),
        "requested_device": args.device,
        "raw_result_path": payload.get("result_path"),
        "synthetic_pkl": payload.get("synthetic_pkl"),
        "train_pkl": payload.get("train_pkl"),
        "validation_pkl": payload.get("threshold_vld_pkl"),
        "target_test_pkl": payload.get("target_test_pkl"),
        "attack_pkl": payload.get("attack_pkl"),
        "TP": payload.get("TP"),
        "TN": payload.get("TN"),
        "FP": payload.get("FP"),
        "FN": payload.get("FN"),
    }


def smartgen_env(scenario: str) -> str:
    if scenario != "st":
        raise ValueError("only scenario=st is supported; it maps to SmartGen env=spring")
    return "spring"


def command_text() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def write_csv(path: Path, metrics: dict[str, Any]) -> None:
    fields = [
        "status",
        "dataset",
        "scenario",
        "variant",
        "seed",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "fpr",
        "fnr",
        "threshold",
        "threshold_source",
        "train_size",
        "validation_size",
        "test_size",
        "generated_size",
        "downstream_pipeline",
        "generator",
        "api_llm",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: metrics.get(field, "") for field in fields})


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "null"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Stage4C SmartGen Built-in AD Run",
        "",
        "| field | value |",
        "| --- | --- |",
    ]
    for key in [
        "status",
        "dataset",
        "scenario",
        "variant",
        "seed",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "fpr",
        "fnr",
        "threshold",
        "threshold_source",
        "train_size",
        "validation_size",
        "test_size",
        "generated_size",
        "downstream_pipeline",
        "generator",
        "api_llm",
    ]:
        lines.append(f"| {key} | {fmt(metrics.get(key))} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def required_paths(args: argparse.Namespace) -> dict[str, Path]:
    env = smartgen_env(args.scenario)
    defaults = default_smartgen_paths(args.smartgen_root, args.dataset, env)
    paths = {
        "generated_pkl": args.generated_pkl,
        "smartgen_root": args.smartgen_root,
        "causal_smart_home_root": args.causal_smart_home_root,
        "attack_pkl": args.attack_pkl or defaults["attack_pkl"],
        "target_test_pkl": args.target_test_pkl or defaults["target_test_pkl"],
    }
    if args.validation_pkl:
        paths["validation_pkl"] = args.validation_pkl
    return {key: path.resolve() for key, path in paths.items()}


def write_failure(out_dir: Path, args: argparse.Namespace, reason: str, missing: list[str] | None = None) -> None:
    report = {
        "status": "failed",
        "command_attempted": command_text(),
        "reason": reason,
        "missing_files": missing or [],
        "stderr_tail": traceback.format_exc(limit=8),
        "dataset": args.dataset,
        "scenario": args.scenario,
        "variant": args.variant,
        "seed": args.seed,
        "downstream_pipeline": "smartgen_builtin_anomaly_detection_pipeline",
    }
    (out_dir / "failure_report.json").write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_command.sh").write_text("#!/usr/bin/env bash\n" + command_text() + "\n", encoding="utf-8")

    paths = required_paths(args)
    missing = [f"{key}: {path}" for key, path in paths.items() if not path.exists()]
    manifest = {
        "dataset": args.dataset,
        "scenario": args.scenario,
        "smartgen_env": smartgen_env(args.scenario),
        "variant": args.variant,
        "seed": args.seed,
        "paths": {key: str(path) for key, path in paths.items()},
    }
    (out_dir / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    config_payload = {
        "downstream_pipeline": "smartgen_builtin_anomaly_detection_pipeline",
        "smartgen_entrypoint": str((args.smartgen_root / "anomaly_detection_pipeline" / "main.py").resolve()),
        "uses_smartguard": False,
        "generator": generator_for_variant(args.variant),
        "api_llm": False,
        "surrogate_for_smartgen_llm": args.variant != "stage3_prompt_only_baseline",
        "epochs": args.epochs,
        "split_ratio": args.split_ratio,
        "device": args.device,
        "threshold_percentage": args.threshold_percentage,
    }
    (out_dir / "config.json").write_text(json.dumps(jsonable(config_payload), ensure_ascii=False, indent=2), encoding="utf-8")

    if missing:
        write_failure(out_dir, args, reason="missing required input file(s)", missing=missing)
        raise FileNotFoundError(f"missing required input file(s): {missing}")

    try:
        config = SmartGenAnomalyRunConfig(
            smartgen_root=args.smartgen_root.resolve(),
            dataset=args.dataset,
            env=smartgen_env(args.scenario),
            synthetic_pkl=args.generated_pkl.resolve(),
            out_dir=out_dir,
            tag=f"{args.dataset}_{args.scenario}_{args.variant}_seed{args.seed}",
            epochs=args.epochs,
            seed=args.seed,
            split_ratio=args.split_ratio,
            device=args.device,
            cuda_visible_devices=args.cuda_visible_devices,
            dry_run=args.dry_run,
            attack_pkl=paths["attack_pkl"],
            target_test_pkl=paths["target_test_pkl"],
            validation_pkl=paths.get("validation_pkl"),
            threshold_percentage=args.threshold_percentage,
        )
        payload = run_smartgen_anomaly_experiment(config)
        payload["generated_size_checked"] = len(load_pickle(args.generated_pkl))
        normalized = normalize_metrics(payload, args)
        (out_dir / "raw_smartgen_metrics.json").write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "normalized_metrics.json").write_text(json.dumps(jsonable(normalized), ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "downstream_ad_metrics.json").write_text(json.dumps(jsonable(normalized), ensure_ascii=False, indent=2), encoding="utf-8")
        write_csv(out_dir / "metrics.csv", normalized)
        write_markdown(out_dir / "metrics.md", normalized)
        print(json.dumps(jsonable(normalized), ensure_ascii=False, indent=2))
    except Exception as exc:
        write_failure(out_dir, args, reason=str(exc), missing=[])
        raise


if __name__ == "__main__":
    main()
