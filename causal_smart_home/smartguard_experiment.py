from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence
import csv
import importlib.util
import json
import os
import pickle
import sys


ATTACKS_DIC = {
    "an": {
        "SD": ["light_attack", "camera_attack", "television_attack"],
        "MD": ["smartlock_attack1", "smartlock_attack2"],
        "DM": ["blind_attack"],
        "DD": ["bathheater_attack"],
    },
    "fr": {
        "SD": ["light_attack", "camera_attack", "television_attack"],
        "MD": ["smartlock1_attack", "smartlock2_attack"],
        "DM": ["airconditioner_attack", "blind_attack"],
        "DD": ["microwave_attack"],
    },
    "sp": {
        "SD": ["light_attack", "camera_attack", "television_attack"],
        "MD": ["smartlock1_attack", "smartlock2_attack"],
        "DM": ["airconditioner_attack", "blind_attack", "watervalve_attack"],
        "DD": ["microwave_attack"],
    },
    "us": {
        "SD": ["light_attack", "camera_attack", "television_attack"],
        "MD": ["smartlock_attack1", "smartlock_attack2"],
        "DM": ["airconditioner_attack", "blind_attack", "watervalve_attack"],
        "DD": ["microwave_attack"],
    },
}


@dataclass(frozen=True)
class SmartGuardRunConfig:
    smartguard_root: Path
    dataset: str
    base_train_pkl: Path
    add_pkl: Path
    out_dir: Path
    tag: str
    vld_pkl: Path | None = None
    test_pkl: Path | None = None
    sequence_length: int | None = None
    pad_value: int = 0
    epochs: int = 60
    threshold_percentage: float = 95.0
    model: str = "SmartGuard"
    mask_strategy: str = "loss_guided"
    mask_ratio: float = 0.2
    mask_step: int = 4
    layer: int = 2
    batch: int = 1024
    embedding: int = 256
    TTPE: bool = True
    LDMS: bool = True
    seed: int = 2023
    attacks: tuple[str, ...] = ("SD", "MD", "DM", "DD")
    dry_run: bool = False


def default_smartguard_paths(smartguard_root: str | Path, dataset: str) -> dict[str, Path]:
    root = Path(smartguard_root).resolve()
    data_dir = root / "data" / f"{dataset}_data"
    return {
        "base_train_pkl": data_dir / f"{dataset}_trn_instance_10.pkl",
        "vld_pkl": data_dir / f"{dataset}_vld_instance_10.pkl",
        "test_pkl": data_dir / f"{dataset}_test_instance_10.pkl",
    }


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: str | Path, obj: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(obj, f)


def normalize_numeric_sequences(
    sequences: Sequence[Sequence[int]],
    sequence_length: int,
    pad_value: int = 0,
) -> list[list[int]]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if sequence_length % 4 != 0:
        raise ValueError("sequence_length must be divisible by 4")
    normalized: list[list[int]] = []
    for seq in sequences:
        row = list(seq)
        if len(row) >= sequence_length:
            normalized.append(row[:sequence_length])
        else:
            normalized.append(row + [pad_value] * (sequence_length - len(row)))
    return normalized


def infer_fixed_sequence_length(sequences: Sequence[Sequence[int]]) -> int:
    lengths = sorted({len(seq) for seq in sequences})
    if len(lengths) != 1:
        raise ValueError(f"cannot infer one fixed sequence length from lengths={lengths}")
    return lengths[0]


def merge_training_sequences(
    base_train_pkl: str | Path,
    add_pkl: str | Path,
    out_pkl: str | Path,
    sequence_length: int | None = None,
    pad_value: int = 0,
) -> dict[str, Any]:
    base = load_pickle(base_train_pkl)
    added = load_pickle(add_pkl)
    target_length = sequence_length or infer_fixed_sequence_length(base)
    merged = normalize_numeric_sequences(base, target_length, pad_value=pad_value)
    merged += normalize_numeric_sequences(added, target_length, pad_value=pad_value)
    save_pickle(out_pkl, merged)
    return {
        "base_train_pkl": str(Path(base_train_pkl).resolve()),
        "add_pkl": str(Path(add_pkl).resolve()),
        "merged_train_pkl": str(Path(out_pkl).resolve()),
        "base_size": len(base),
        "added_size": len(added),
        "merged_size": len(merged),
        "sequence_length": target_length,
        "pad_value": pad_value,
    }


def resolve_sweep_rows(
    sweep_summary_csv: str | Path,
    slugs: Sequence[str] | None = None,
    cwd: str | Path | None = None,
) -> list[dict[str, str]]:
    summary = Path(sweep_summary_csv).resolve()
    selected = set(slugs or [])
    with open(summary, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if selected:
        rows = [row for row in rows if row.get("slug") in selected]
    if selected and len(rows) != len(selected):
        found = {row.get("slug") for row in rows}
        missing = sorted(selected - found)
        raise ValueError(f"sweep summary is missing selected slugs: {missing}")

    root = Path(cwd).resolve() if cwd else Path.cwd()
    resolved: list[dict[str, str]] = []
    for row in rows:
        kept = row.get("kept_path", "")
        if not kept:
            raise ValueError(f"row {row.get('slug')} has no kept_path")
        kept_path = Path(kept)
        if not kept_path.is_absolute():
            candidate = root / kept_path
            kept_path = candidate if candidate.exists() else kept_path.resolve()
        item = dict(row)
        item["kept_path"] = str(kept_path)
        resolved.append(item)
    return resolved


def aggregate_attack_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(r.get("TP", 0)) for r in results)
    tn = sum(int(r.get("TN", 0)) for r in results)
    fp = sum(int(r.get("FP", 0)) for r in results)
    fn = sum(int(r.get("FN", 0)) for r in results)
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "recall": recall,
        "precision": precision,
        "accuracy": accuracy,
        "f1_score": f1,
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _import_smartguard_train(smartguard_root: Path):
    train_path = smartguard_root / "train.py"
    if not train_path.exists():
        raise FileNotFoundError(f"SmartGuard train.py not found: {train_path}")
    root_str = str(smartguard_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    spec = importlib.util.spec_from_file_location("_csh_smartguard_train", train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import SmartGuard train module from {train_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_work_data_link(work_dir: Path, smartguard_root: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    link = work_dir / "data"
    target = smartguard_root / "data"
    if link.exists() or link.is_symlink():
        return
    link.symlink_to(target, target_is_directory=True)


class _Cwd:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.previous: str | None = None

    def __enter__(self):
        self.previous = os.getcwd()
        os.chdir(self.path)

    def __exit__(self, exc_type, exc, tb):
        if self.previous is not None:
            os.chdir(self.previous)


def _build_args(config: SmartGuardRunConfig) -> SimpleNamespace:
    return SimpleNamespace(
        epochs=config.epochs,
        model=config.model,
        dataset=config.dataset,
        mask_strategy=config.mask_strategy,
        mask_ratio=config.mask_ratio,
        mask_step=config.mask_step,
        layer=config.layer,
        batch=config.batch,
        embedding=config.embedding,
        TTPE=config.TTPE,
        LDMS=config.LDMS,
    )


def _configure_train_module(train_module, config: SmartGuardRunConfig, train_pkl: Path, model_path: Path) -> Any:
    args = _build_args(config)
    vld_pkl = config.vld_pkl or default_smartguard_paths(config.smartguard_root, config.dataset)["vld_pkl"]
    test_pkl = config.test_pkl or default_smartguard_paths(config.smartguard_root, config.dataset)["test_pkl"]
    if config.dataset not in train_module.vocab_dic:
        raise ValueError(f"SmartGuard train.py does not define vocab size for dataset={config.dataset}")
    train_module.args = args
    train_module.vocab_size = train_module.vocab_dic[config.dataset]
    train_module.train_file1 = str(train_pkl)
    train_module.train_file2 = str(train_pkl)
    train_module.vld_file = str(vld_pkl)
    train_module.test_file2 = str(test_pkl)
    train_module.batch_size = config.batch
    train_module.model_name = str(model_path)
    train_module.attacks_dic = ATTACKS_DIC
    train_module.model = train_module.SmartGuard(
        vocab_size=train_module.vocab_size,
        d_model=config.embedding,
        nhead=8,
        num_layers=config.layer,
        mask_strategy=config.mask_strategy,
        mask_ratio=config.mask_ratio,
        mask_step=config.mask_step,
        TTPE_flag=config.TTPE,
    )
    return args


def run_smartguard_experiment(config: SmartGuardRunConfig) -> dict[str, Any]:
    out_dir = config.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_train = out_dir / f"{config.tag}_merged_train.pkl"
    model_path = out_dir / f"{config.tag}_smartguard.pth"
    result_path = out_dir / f"{config.tag}_smartguard_eval.json"
    work_dir = out_dir / "_smartguard_work"

    merge_info = merge_training_sequences(
        config.base_train_pkl,
        config.add_pkl,
        merged_train,
        sequence_length=config.sequence_length,
        pad_value=config.pad_value,
    )
    payload: dict[str, Any] = {
        "tag": config.tag,
        "dataset": config.dataset,
        "dry_run": config.dry_run,
        "merge": merge_info,
        "model_path": str(model_path),
        "result_path": str(result_path),
        "epochs": config.epochs,
        "threshold_percentage": config.threshold_percentage,
        "attacks": list(config.attacks),
    }
    if config.dry_run:
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    _ensure_work_data_link(work_dir, config.smartguard_root)
    train_module = _import_smartguard_train(config.smartguard_root)
    args = _configure_train_module(train_module, config, merged_train, model_path)

    with _Cwd(work_dir):
        train_module.setup_seed(config.seed)
        train_module.train(args)
        threshold = train_module.find_threshold(percentage=config.threshold_percentage)
        weights = train_module.get_behavior_weight()
        results = []
        for attack_type in config.attacks:
            results.append(train_module.evaluate(threshold, weights, attack_type))

    payload.update(
        {
            "threshold": float(threshold),
            "weights_count": len(weights),
            "results": _jsonable(results),
            "aggregate": aggregate_attack_results(results),
        }
    )
    result_path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


SWEEP_EVAL_CSV_FIELDS = [
    "tag",
    "slug",
    "kept_path",
    "merged_train_pkl",
    "base_size",
    "added_size",
    "merged_size",
    "sequence_length",
    "epochs",
    "threshold_percentage",
    "TP",
    "TN",
    "FP",
    "FN",
    "recall",
    "precision",
    "accuracy",
    "f1_score",
    "result_path",
    "model_path",
]


def summarize_smartguard_payload(payload: dict[str, Any], slug: str, kept_path: str) -> dict[str, Any]:
    merge = payload.get("merge", {})
    aggregate = payload.get("aggregate", {})
    return {
        "tag": payload.get("tag", ""),
        "slug": slug,
        "kept_path": kept_path,
        "merged_train_pkl": merge.get("merged_train_pkl", ""),
        "base_size": merge.get("base_size", ""),
        "added_size": merge.get("added_size", ""),
        "merged_size": merge.get("merged_size", ""),
        "sequence_length": merge.get("sequence_length", ""),
        "epochs": payload.get("epochs", ""),
        "threshold_percentage": payload.get("threshold_percentage", ""),
        "TP": aggregate.get("TP", ""),
        "TN": aggregate.get("TN", ""),
        "FP": aggregate.get("FP", ""),
        "FN": aggregate.get("FN", ""),
        "recall": aggregate.get("recall", ""),
        "precision": aggregate.get("precision", ""),
        "accuracy": aggregate.get("accuracy", ""),
        "f1_score": aggregate.get("f1_score", ""),
        "result_path": payload.get("result_path", ""),
        "model_path": payload.get("model_path", ""),
    }


def write_smartguard_sweep_summary(rows: Sequence[dict[str, Any]], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "smartguard_sweep_eval_summary.csv"
    json_path = out / "smartguard_sweep_eval_summary.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_EVAL_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SWEEP_EVAL_CSV_FIELDS})
    json_path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def run_smartguard_sweep(
    sweep_summary_csv: str | Path,
    base_config: SmartGuardRunConfig,
    slugs: Sequence[str],
) -> list[dict[str, Any]]:
    rows = resolve_sweep_rows(sweep_summary_csv, slugs=slugs)
    summary_rows: list[dict[str, Any]] = []
    for row in rows:
        slug = row["slug"]
        tag = f"{base_config.tag}_{slug}" if base_config.tag else slug
        run_config = SmartGuardRunConfig(
            smartguard_root=base_config.smartguard_root,
            dataset=base_config.dataset,
            base_train_pkl=base_config.base_train_pkl,
            add_pkl=Path(row["kept_path"]),
            out_dir=base_config.out_dir,
            tag=tag,
            vld_pkl=base_config.vld_pkl,
            test_pkl=base_config.test_pkl,
            sequence_length=base_config.sequence_length,
            pad_value=base_config.pad_value,
            epochs=base_config.epochs,
            threshold_percentage=base_config.threshold_percentage,
            model=base_config.model,
            mask_strategy=base_config.mask_strategy,
            mask_ratio=base_config.mask_ratio,
            mask_step=base_config.mask_step,
            layer=base_config.layer,
            batch=base_config.batch,
            embedding=base_config.embedding,
            TTPE=base_config.TTPE,
            LDMS=base_config.LDMS,
            seed=base_config.seed,
            attacks=base_config.attacks,
            dry_run=base_config.dry_run,
        )
        payload = run_smartguard_experiment(run_config)
        summary_rows.append(summarize_smartguard_payload(payload, slug=slug, kept_path=row["kept_path"]))
    write_smartguard_sweep_summary(summary_rows, base_config.out_dir)
    return summary_rows
