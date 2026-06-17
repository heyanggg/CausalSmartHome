from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import csv
import importlib.util
import json
import pickle
import random
import sys

from .smartguard_experiment import resolve_sweep_rows


VOCAB_DIC = {"fr": 223, "sp": 235, "us": 269}

DEFAULT_THRESHOLDS = {
    ("fr", "spring"): "0.918",
    ("fr", "night"): "0.92",
    ("fr", "multiple"): "0.915",
    ("sp", "spring"): "0.915",
    ("sp", "night"): "0.917",
    ("sp", "multiple"): "0.915",
    ("us", "spring"): "0.905",
    ("us", "night"): "0.919",
    ("us", "multiple"): "0.913",
}

DEFAULT_THRESHOLD_PERCENTAGES = {
    ("fr", "spring"): 95.5,
    ("fr", "night"): 95.0,
    ("fr", "multiple"): 99.0,
    ("sp", "spring"): 95.0,
    ("sp", "night"): 95.0,
    ("sp", "multiple"): 99.0,
    ("us", "spring"): 95.0,
    ("us", "night"): 93.0,
    ("us", "multiple"): 99.0,
}


@dataclass(frozen=True)
class SmartGenAnomalyRunConfig:
    smartgen_root: Path
    dataset: str
    env: str
    synthetic_pkl: Path
    out_dir: Path
    tag: str
    threshold: str | None = None
    threshold_percentage: float | None = None
    method: str = "SPPC"
    model: str = "gpt-4o"
    epochs: int = 15
    seed: int = 2024
    split_ratio: float = 0.8
    dry_run: bool = False
    attack_pkl: Path | None = None
    target_test_pkl: Path | None = None


def default_smartgen_paths(smartgen_root: str | Path, dataset: str, env: str) -> dict[str, Path]:
    pipeline = Path(smartgen_root).resolve() / "anomaly_detection_pipeline"
    if env == "spring":
        attack_name = f"labeled_{dataset}_spring_attack_heater.pkl"
    elif env == "night":
        attack_name = f"labeled_{dataset}_night_attack_time.pkl"
    elif env == "multiple":
        attack_name = f"labeled_{dataset}_multiple_attack_tv.pkl"
    else:
        raise ValueError("env must be spring, night, or multiple")
    return {
        "pipeline_root": pipeline,
        "attack_pkl": pipeline / "attack" / dataset / attack_name,
        "target_test_pkl": pipeline / "test" / dataset / env / "test.pkl",
    }


def default_synthetic_pkl(
    smartgen_root: str | Path,
    dataset: str,
    env: str,
    method: str = "SPPC",
    model: str = "gpt-4o",
    threshold: str | None = None,
) -> Path:
    threshold = threshold or DEFAULT_THRESHOLDS[(dataset, env)]
    return (
        Path(smartgen_root).resolve()
        / "anomaly_detection_pipeline"
        / "synthetic_data"
        / f"{dataset}_{env}_generation_{method}_th={threshold}_{model}_seq_filter_true.pkl"
    )


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: str | Path, obj: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(obj, f)


def split_random_to_files(
    data_file: str | Path,
    train_file: str | Path,
    vld_file: str | Path,
    split_ratio: float = 0.8,
    seed: int = 2024,
) -> tuple[list[Any], list[Any]]:
    data = list(load_pickle(data_file))
    rng = random.Random(seed)
    rng.shuffle(data)
    split_index = int(len(data) * split_ratio)
    train = data[:split_index]
    vld = data[split_index:]
    save_pickle(train_file, train)
    save_pickle(vld_file, vld)
    return train, vld


def _import_smartgen_anomaly_module(smartgen_root: Path):
    pipeline_root = smartgen_root / "anomaly_detection_pipeline"
    module_path = pipeline_root / "Anomaly_Detection_pipeline_model.py"
    if not module_path.exists():
        raise FileNotFoundError(f"SmartGen anomaly module not found: {module_path}")
    root_str = str(pipeline_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    spec = importlib.util.spec_from_file_location("_csh_smartgen_anomaly", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import SmartGen anomaly module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_cuda_available() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("SmartGen anomaly evaluation requires torch for real training runs") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "SmartGen anomaly_detection_pipeline calls .cuda() directly; run this command in a CUDA-enabled "
            "environment, or patch SmartGen for CPU/GPU adaptive training."
        )


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _prepare_split(config: SmartGenAnomalyRunConfig, train_pkl: Path, vld_pkl: Path) -> tuple[int, int]:
    if config.env == "multiple":
        data = load_pickle(config.synthetic_pkl)
        save_pickle(train_pkl, data)
        save_pickle(vld_pkl, data)
        return len(data), len(data)
    train, vld = split_random_to_files(
        config.synthetic_pkl,
        train_pkl,
        vld_pkl,
        split_ratio=config.split_ratio,
        seed=config.seed,
    )
    return len(train), len(vld)


def run_smartgen_anomaly_experiment(config: SmartGenAnomalyRunConfig) -> dict[str, Any]:
    smartgen_root = config.smartgen_root.resolve()
    defaults = default_smartgen_paths(smartgen_root, config.dataset, config.env)
    threshold = config.threshold or DEFAULT_THRESHOLDS[(config.dataset, config.env)]
    threshold_percentage = (
        config.threshold_percentage
        if config.threshold_percentage is not None
        else DEFAULT_THRESHOLD_PERCENTAGES[(config.dataset, config.env)]
    )
    out_dir = config.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_pkl = out_dir / f"{config.tag}_train.pkl"
    vld_pkl = out_dir / f"{config.tag}_vld.pkl"
    model_path = out_dir / f"{config.tag}_transformer_autoencoder.pth"
    result_path = out_dir / f"{config.tag}_smartgen_anomaly_eval.json"

    synthetic_data = load_pickle(config.synthetic_pkl)
    train_size, vld_size = _prepare_split(config, train_pkl, vld_pkl)

    payload: dict[str, Any] = {
        "tag": config.tag,
        "dataset": config.dataset,
        "env": config.env,
        "method": config.method,
        "model": config.model,
        "threshold": threshold,
        "threshold_percentage": threshold_percentage,
        "epochs": config.epochs,
        "split_ratio": config.split_ratio,
        "dry_run": config.dry_run,
        "synthetic_pkl": str(config.synthetic_pkl.resolve()),
        "synthetic_size": len(synthetic_data),
        "train_pkl": str(train_pkl),
        "vld_pkl": str(vld_pkl),
        "train_size": train_size,
        "vld_size": vld_size,
        "model_path": str(model_path),
        "result_path": str(result_path),
        "attack_pkl": str((config.attack_pkl or defaults["attack_pkl"]).resolve()),
        "target_test_pkl": str((config.target_test_pkl or defaults["target_test_pkl"]).resolve()),
    }
    if config.dry_run:
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    _ensure_cuda_available()
    module = _import_smartgen_anomaly_module(smartgen_root)
    vocab_size = getattr(module, "vocab_dic", VOCAB_DIC)[config.dataset]
    seq_len = 10
    module.setup_seed(config.seed)
    module.train(config.env, vocab_size, config.epochs, str(train_pkl), str(model_path), seq_len)
    learned_threshold = module.find_threshold(
        config.env,
        vocab_size,
        str(vld_pkl),
        str(model_path),
        seq_len,
        percentage=threshold_percentage,
    )
    recall, precision, accuracy, f1_score = module.evaluate(
        config.env,
        vocab_size,
        str(config.attack_pkl or defaults["attack_pkl"]),
        str(config.target_test_pkl or defaults["target_test_pkl"]),
        str(model_path),
        seq_len,
        threshold=learned_threshold,
    )
    payload.update(
        {
            "learned_threshold": float(learned_threshold),
            "recall": float(recall),
            "precision": float(precision),
            "accuracy": float(accuracy),
            "F1 score": float(f1_score),
        }
    )
    result_path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


SMARTGEN_SWEEP_FIELDS = [
    "tag",
    "slug",
    "synthetic_pkl",
    "synthetic_size",
    "train_size",
    "vld_size",
    "recall",
    "precision",
    "accuracy",
    "F1 score",
    "learned_threshold",
    "result_path",
    "model_path",
]


def summarize_smartgen_payload(payload: dict[str, Any], slug: str) -> dict[str, Any]:
    return {
        "tag": payload.get("tag", ""),
        "slug": slug,
        "synthetic_pkl": payload.get("synthetic_pkl", ""),
        "synthetic_size": payload.get("synthetic_size", ""),
        "train_size": payload.get("train_size", ""),
        "vld_size": payload.get("vld_size", ""),
        "recall": payload.get("recall", ""),
        "precision": payload.get("precision", ""),
        "accuracy": payload.get("accuracy", ""),
        "F1 score": payload.get("F1 score", ""),
        "learned_threshold": payload.get("learned_threshold", ""),
        "result_path": payload.get("result_path", ""),
        "model_path": payload.get("model_path", ""),
    }


def write_smartgen_sweep_summary(rows: Sequence[dict[str, Any]], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "smartgen_anomaly_sweep_summary.csv"
    json_path = out / "smartgen_anomaly_sweep_summary.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SMARTGEN_SWEEP_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SMARTGEN_SWEEP_FIELDS})
    json_path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def run_smartgen_anomaly_sweep(
    sweep_summary_csv: str | Path,
    base_config: SmartGenAnomalyRunConfig,
    slugs: Sequence[str],
) -> list[dict[str, Any]]:
    rows = resolve_sweep_rows(sweep_summary_csv, slugs=slugs)
    summary_rows: list[dict[str, Any]] = []
    for row in rows:
        slug = row["slug"]
        tag = f"{base_config.tag}_{slug}" if base_config.tag else slug
        config = SmartGenAnomalyRunConfig(
            smartgen_root=base_config.smartgen_root,
            dataset=base_config.dataset,
            env=base_config.env,
            synthetic_pkl=Path(row["kept_path"]),
            out_dir=base_config.out_dir,
            tag=tag,
            threshold=base_config.threshold,
            threshold_percentage=base_config.threshold_percentage,
            method=base_config.method,
            model=base_config.model,
            epochs=base_config.epochs,
            seed=base_config.seed,
            split_ratio=base_config.split_ratio,
            dry_run=base_config.dry_run,
            attack_pkl=base_config.attack_pkl,
            target_test_pkl=base_config.target_test_pkl,
        )
        payload = run_smartgen_anomaly_experiment(config)
        summary_rows.append(summarize_smartgen_payload(payload, slug=slug))
    write_smartgen_sweep_summary(summary_rows, base_config.out_dir)
    return summary_rows
