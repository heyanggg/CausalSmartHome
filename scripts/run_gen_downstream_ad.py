#!/usr/bin/env python
"""Gen 内置下游异常检测器的命令行包装器。

真正的 Gen AD 协议实现在 ``gen_downstream_ad.py``。本脚本在其外层补充实验
溯源、标准化 metrics、输入 manifest 和失败报告。
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.gen_downstream_ad import (
    DATASETS,
    DEFAULT_THRESHOLD_PERCENTAGES,
    ENV_BY_SCENARIO,
    GenDownstreamADRunConfig,
    default_gen_paths,
    env_for_scenario,
    load_pickle,
    run_gen_downstream_ad_experiment,
)
from causal_smart_home.experiment_paths import GEN_ROOT
from causal_smart_home.json_utils import jsonable

PROPOSED_VARIANT = "proposed_zero_target_causal_gss_codex"
VARIANTS = {
    PROPOSED_VARIANT,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gen built-in downstream AD for the current main experiment.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--scenario", required=True, choices=sorted(ENV_BY_SCENARIO), help="Gen target context: st/spring, tt/night, or nt/multiple.")
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--generated-pkl", required=True, type=Path, help="Input pkl for this AD variant.")
    parser.add_argument("--pre-tof-pkl", type=Path, help="Pre-TOF generated pkl used only for provenance/counts.")
    parser.add_argument("--gen-tof-pkl", type=Path, help="Gen original TOF output pkl used for provenance/counts.")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--gen-root", type=Path, default=GEN_ROOT)
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


def _pickle_len(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        return len(load_pickle(path))
    except Exception:
        return None


def _read_tof_report(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    candidates = [
        path.parent / "gen_original_tof_report.json",
        path.with_suffix(path.suffix + ".gen_original_tof_report.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


def generated_counts(args: argparse.Namespace) -> dict[str, Any]:
    """推断 pre-TOF、post-Gen-TOF、post-Causal-TOF 的样本数量用于报告。"""
    variant = args.variant
    current_len = _pickle_len(args.generated_pkl)
    pre_tof_len = _pickle_len(args.pre_tof_pkl)
    gen_tof_len = _pickle_len(args.gen_tof_pkl)
    tof_report = _read_tof_report(args.gen_tof_pkl or (args.generated_pkl if used_gen_original_tof_for_variant(variant) else None))

    before = pre_tof_len
    after_gen = gen_tof_len
    if variant == PROPOSED_VARIANT:
        before = before if before is not None else tof_report.get("num_generated_before_tof")
        after_gen = current_len

    return {
        "input_pkl": str(args.generated_pkl.resolve()),
        "input_stage": input_stage_for_variant(variant),
        "used_gen_original_tof": used_gen_original_tof_for_variant(variant),
        "num_generated_before_tof": before,
        "num_generated_after_gen_tof": after_gen,
    }


def f1(payload: dict[str, Any]) -> float | None:
    value = payload.get("F1 score", payload.get("F1"))
    return float(value) if value is not None else None


def generator_for_variant(variant: str) -> str:
    return "codex_generation"


def read_generation_provenance(path: Path | None) -> dict[str, Any]:
    """读取 generated pkl 旁边的 generator/model 溯源信息。"""
    if path is None:
        return {}
    candidates = [
        path.parent / "generation_report.json",
        path.with_suffix(path.suffix + ".generation_report.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            return {
                key: payload[key]
                for key in (
                    "generator",
                    "generation_model",
                    "manual_generation",
                )
                if key in payload
            }
    return {}


def input_stage_for_variant(variant: str) -> str:
    if variant == PROPOSED_VARIANT:
        return "gen_original_tof"
    return variant


def used_gen_original_tof_for_variant(variant: str) -> bool:
    return variant == PROPOSED_VARIANT


def normalize_metrics(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """把 raw Gen AD payload 展平为汇总脚本使用的 per-seed schema。"""
    variant = args.variant
    provenance = read_generation_provenance(args.pre_tof_pkl or args.generated_pkl)
    threshold_percentage = (
        args.threshold_percentage
        if args.threshold_percentage is not None
        else DEFAULT_THRESHOLD_PERCENTAGES[(args.dataset, gen_env(args.scenario))]
    )
    normalized = {
        "status": "dry_run" if payload.get("dry_run") else "success",
        "dataset": args.dataset,
        "scenario": args.scenario,
        "seed": args.seed,
        "variant": variant,
        "gen_env": gen_env(args.scenario),
        "downstream_pipeline": "gen_builtin_downstream_ad",
        "generator": provenance.get("generator", generator_for_variant(variant)),
        "generation_model": provenance.get("generation_model", "Codex"),
        "manual_generation": provenance.get("manual_generation", True),
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
        "epochs": args.epochs,
        "split_ratio": args.split_ratio,
        "device": payload.get("device"),
        "requested_device": args.device,
        "pipeline_result_path": payload.get("result_path"),
        "metrics_path": str((args.out_dir.resolve() / "downstream_ad_metrics.json")),
        "run_dir": str(args.out_dir.resolve()),
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
    normalized.update(generated_counts(args))
    return normalized


def gen_env(scenario: str) -> str:
    return env_for_scenario(scenario)


def command_text() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


PER_SEED_FIELDS = [
    "dataset",
    "scenario",
    "seed",
    "variant",
    "input_pkl",
    "input_stage",
    "used_gen_original_tof",
    "downstream_pipeline",
    "generator",
    "generation_model",
    "num_generated_before_tof",
    "num_generated_after_gen_tof",
    "train_size",
    "validation_size",
    "test_size",
    "threshold",
    "threshold_source",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "fpr",
    "fnr",
    "status",
    "run_dir",
    "metrics_path",
]


def write_csv(path: Path, metrics: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_SEED_FIELDS)
        writer.writeheader()
        writer.writerow({field: metrics.get(field, "") for field in PER_SEED_FIELDS})


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "null"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Gen Built-in Downstream AD Run",
        "",
        "| field | value |",
        "| --- | --- |",
    ]
    for key in PER_SEED_FIELDS:
        lines.append(f"| {key} | {fmt(metrics.get(key))} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def required_paths(args: argparse.Namespace) -> dict[str, Path]:
    """在昂贵训练开始前列出并检查必须存在的输入文件。"""
    env = gen_env(args.scenario)
    defaults = default_gen_paths(args.gen_root, args.dataset, env)
    paths = {
        "generated_pkl": args.generated_pkl,
        "gen_root": args.gen_root,
        "attack_pkl": args.attack_pkl or defaults["attack_pkl"],
        "target_test_pkl": args.target_test_pkl or defaults["target_test_pkl"],
    }
    if args.validation_pkl:
        paths["validation_pkl"] = args.validation_pkl
    return {key: path.resolve() for key, path in paths.items()}


def write_failure(out_dir: Path, args: argparse.Namespace, reason: str, missing: list[str] | None = None) -> None:
    """保存足够上下文，方便排查失败的 AD 运行。"""
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
        "downstream_pipeline": "gen_builtin_downstream_ad",
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
        "gen_env": gen_env(args.scenario),
        "variant": args.variant,
        "seed": args.seed,
        "paths": {key: str(path) for key, path in paths.items()},
    }
    (out_dir / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    provenance = read_generation_provenance(args.pre_tof_pkl or args.generated_pkl)
    variant = args.variant
    config_payload = {
        "downstream_pipeline": "gen_builtin_downstream_ad",
        "gen_entrypoint": str((args.gen_root / "anomaly_detection_pipeline" / "models1.py").resolve()),
        "generator": provenance.get("generator", generator_for_variant(variant)),
        "generation_model": provenance.get("generation_model", "Codex"),
        "manual_generation": provenance.get("manual_generation", True),
        "input_stage": input_stage_for_variant(variant),
        "used_gen_original_tof": used_gen_original_tof_for_variant(variant),
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
        config = GenDownstreamADRunConfig(
            gen_root=args.gen_root.resolve(),
            dataset=args.dataset,
            env=gen_env(args.scenario),
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
        payload = run_gen_downstream_ad_experiment(config)
        payload["generated_size_checked"] = len(load_pickle(args.generated_pkl))
        normalized = normalize_metrics(payload, args)
        (out_dir / "gen_downstream_ad_payload.json").write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
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
