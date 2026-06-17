from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import csv
import importlib.util
import json
import math
import numpy as np
import os
import pickle
import random
import sys

from .smartguard_experiment import resolve_sweep_rows
from .causal_filter import CausalConsistencyFilter
from .causal_prior import CausalPrior
from .schema import load_numeric_sequences


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
    device: str = "cuda"
    cuda_visible_devices: str | None = "0"
    dry_run: bool = False
    attack_pkl: Path | None = None
    target_test_pkl: Path | None = None
    validation_pkl: Path | None = None
    weight_prior_json: Path | None = None
    weight_top_k_edges: int = 30
    weight_min_edge_weight: float | None = None
    weight_floor: float = 0.2
    weight_power: float = 1.0


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


def _import_smartgen_models(smartgen_root: Path):
    pipeline_root = smartgen_root / "anomaly_detection_pipeline"
    module_path = pipeline_root / "models1.py"
    if not module_path.exists():
        raise FileNotFoundError(f"SmartGen models module not found: {module_path}")
    root_str = str(pipeline_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    spec = importlib.util.spec_from_file_location("_csh_smartgen_models", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import SmartGen models from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_torch_device(requested: str):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("SmartGen anomaly evaluation requires torch for real training runs") from exc
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device, but torch.cuda.is_available() is false")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return torch.device(requested)


def _setup_torch_seed(seed: int) -> None:
    import os
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _pad_sequences(vocab_size: int, sequences: Sequence[Sequence[int]], length: int = 40) -> list[list[int]]:
    padded: list[list[int]] = []
    for seq in sequences:
        row = list(seq)
        if len(row) < length:
            row = row + [vocab_size - 1] * (length - len(row))
        elif len(row) > length:
            row = row[:length]
        padded.append(row)
    return padded


def _dataset_for_env(models_module, env: str):
    if env == "spring":
        return models_module.TimeSeriesDataset2
    if env == "night":
        return models_module.TimeSeriesDataset3
    if env == "multiple":
        return models_module.TimeSeriesDataset4
    raise ValueError("env must be spring, night, or multiple")


def _make_loader(models_module, env: str, vocab_size: int, data_file: str | Path, batch_size: int):
    from torch.utils.data import DataLoader

    data = np.array(_pad_sequences(vocab_size, load_pickle(data_file)))
    dataset = _dataset_for_env(models_module, env)(vocab_size, data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _make_weighted_loader(
    models_module,
    env: str,
    vocab_size: int,
    data_file: str | Path,
    weights_file: str | Path,
    batch_size: int,
):
    from torch.utils.data import DataLoader, Dataset

    data = np.array(_pad_sequences(vocab_size, load_pickle(data_file)))
    dataset = _dataset_for_env(models_module, env)(vocab_size, data)
    weights = np.asarray(load_pickle(weights_file), dtype=np.float32)
    if len(weights) != len(dataset):
        raise ValueError(f"weights length {len(weights)} does not match dataset length {len(dataset)}")

    class _WeightedDataset(Dataset):
        def __len__(self):
            return len(dataset)

        def __getitem__(self, index):
            src, padding_mask, mask_v = dataset[index]
            return src, padding_mask, mask_v, weights[index]

    return DataLoader(_WeightedDataset(), batch_size=batch_size, shuffle=False)


def _sequence_loss(output, src, mask_v, vocab_size: int, seq_len: int, criterion):
    import torch

    loss = criterion(output.view(-1, vocab_size), src.view(-1))
    loss = loss.reshape(-1, seq_len) * mask_v
    denom = torch.sum(mask_v)
    if denom.item() == 0:
        return torch.sum(loss)
    return torch.sum(loss) / denom


def _sequence_loss_per_sample(output, src, mask_v, vocab_size: int, seq_len: int, criterion):
    import torch

    loss = criterion(output.view(-1, vocab_size), src.view(-1))
    loss = loss.reshape(-1, seq_len) * mask_v
    denom = torch.sum(mask_v, dim=1).clamp_min(1).to(loss.dtype)
    return torch.sum(loss, dim=1) / denom


def _weighted_sequence_loss(output, src, mask_v, sample_weights, vocab_size: int, seq_len: int, criterion):
    import torch

    per_sample = _sequence_loss_per_sample(output, src, mask_v, vocab_size, seq_len, criterion)
    weights = sample_weights.to(per_sample.device).to(per_sample.dtype)
    denom = torch.sum(weights).clamp_min(1e-8)
    return torch.sum(per_sample * weights) / denom


def _train_adaptive(
    models_module,
    env: str,
    vocab_size: int,
    epochs: int,
    train_file: str | Path,
    model_path: str | Path,
    seq_len: int,
    device,
    train_weights_file: str | Path | None = None,
) -> list[dict[str, float]]:
    import torch
    import torch.nn as nn
    from torch import optim

    model = models_module.TransformerAutoencoder(
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
    ).to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    if train_weights_file is None:
        train_loader = _make_loader(models_module, env, vocab_size, train_file, batch_size=32)
    else:
        train_loader = _make_weighted_loader(models_module, env, vocab_size, train_file, train_weights_file, batch_size=32)

    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            if train_weights_file is None:
                src, padding_mask, mask_v = batch
                sample_weights = None
            else:
                src, padding_mask, mask_v, sample_weights = batch
            src = src.to(device).long()
            mask_v = mask_v.to(device)
            padding_mask = padding_mask.to(device)
            output = model(src, src_key_padding_mask=padding_mask)
            if sample_weights is None:
                loss = _sequence_loss(output, src, mask_v, vocab_size, seq_len, criterion)
            else:
                loss = _weighted_sequence_loss(output, src, mask_v, sample_weights, vocab_size, seq_len, criterion)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), model_path)
        avg_loss = total_loss / len(train_loader) if len(train_loader) else math.nan
        history.append({"epoch": epoch + 1, "train_loss": avg_loss})
        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_loss:.4f}")
    print("Finished Training")
    return history


def _load_transformer(models_module, vocab_size: int, model_path: str | Path, device):
    import torch

    model = models_module.TransformerAutoencoder(
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def _losses_for_file(
    models_module,
    env: str,
    vocab_size: int,
    data_file: str | Path,
    model_path: str | Path,
    seq_len: int,
    device,
) -> list[float]:
    import torch
    import torch.nn as nn

    loader = _make_loader(models_module, env, vocab_size, data_file, batch_size=1)
    model = _load_transformer(models_module, vocab_size, model_path, device)
    criterion = nn.CrossEntropyLoss(reduction="none")
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            src, padding_mask, mask_v = batch
            src = src.to(device).long()
            mask_v = mask_v.to(device)
            padding_mask = padding_mask.to(device)
            output = model(src, src_key_padding_mask=padding_mask)
            loss = _sequence_loss(output, src, mask_v, vocab_size, seq_len, criterion)
            losses.append(loss.item())
    return losses


def _find_threshold_adaptive(
    models_module,
    env: str,
    vocab_size: int,
    vld_file: str | Path,
    model_path: str | Path,
    seq_len: int,
    percentage: float,
    device,
):
    losses = _losses_for_file(models_module, env, vocab_size, vld_file, model_path, seq_len, device)
    avg_loss = float(np.mean(losses)) if losses else math.nan
    print(f"Avg Loss (Validation Dataset): {avg_loss:.4f}")
    threshold = float(np.percentile(losses, percentage)) if losses else math.nan
    print(f"Percentage:{percentage}% Threshold: {threshold}")
    return threshold, losses


def _evaluate_adaptive(
    models_module,
    env: str,
    vocab_size: int,
    attack_file: str | Path,
    target_test_file: str | Path,
    model_path: str | Path,
    seq_len: int,
    threshold: float,
    device,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    attack_samples = load_pickle(attack_file)
    normal_samples = [(row, 0) for row in _pad_sequences(vocab_size, load_pickle(target_test_file))]
    samples = normal_samples + attack_samples
    data = np.array(_pad_sequences(vocab_size, [item[0] for item in samples]))
    labels = [int(item[1]) for item in samples]
    dataset = _dataset_for_env(models_module, env)(vocab_size, data)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = _load_transformer(models_module, vocab_size, model_path, device)
    criterion = nn.CrossEntropyLoss(reduction="none")

    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            src, padding_mask, mask_v = batch
            src = src.to(device).long()
            mask_v = mask_v.to(device)
            padding_mask = padding_mask.to(device)
            output = model(src, src_key_padding_mask=padding_mask)
            loss = _sequence_loss(output, src, mask_v, vocab_size, seq_len, criterion)
            losses.append(loss.item())

    predictions = [0 if loss < threshold else 1 for loss in losses]
    tp = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 0)
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print(f"Avg Loss (Test Dataset): {float(np.mean(losses)) if losses else math.nan:.4f}")
    print("Recall:", recall)
    print("Precision:", precision)
    print("Accuracy:", accuracy)
    print("F1 Score:", f1_score)
    if losses:
        print(f"Max loss {max(losses):.4f} , Min loss: {min(losses):.4f}")
    print("Finished Test")

    return {
        "recall": recall,
        "precision": precision,
        "accuracy": accuracy,
        "F1 score": f1_score,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "test_size": len(labels),
        "normal_test_size": len(normal_samples),
        "attack_test_size": len(attack_samples),
        "test_loss_avg": float(np.mean(losses)) if losses else math.nan,
        "test_loss_min": min(losses) if losses else math.nan,
        "test_loss_max": max(losses) if losses else math.nan,
    }


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


def _causal_training_weights(
    sequences: Sequence[Sequence[int]],
    prior_json: str | Path,
    top_k_edges: int,
    min_edge_weight: float | None,
    weight_floor: float,
    weight_power: float,
) -> tuple[list[float], list[dict[str, Any]]]:
    if not 0.0 <= weight_floor <= 1.0:
        raise ValueError("weight_floor must be between 0 and 1")
    if weight_power <= 0:
        raise ValueError("weight_power must be positive")
    prior = CausalPrior.load(prior_json)
    scorer = CausalConsistencyFilter(prior, top_k_edges=top_k_edges, min_edge_weight=min_edge_weight)
    weights: list[float] = []
    scores: list[dict[str, Any]] = []
    for score in (scorer.score_sequence(seq) for seq in load_numeric_sequences(sequences)):
        coverage = float(score["causal_coverage"])
        weight = float(weight_floor + (1.0 - weight_floor) * (coverage ** weight_power))
        score = dict(score)
        score["sample_weight"] = weight
        weights.append(weight)
        scores.append(score)
    return weights, scores


def _prepare_training_weights(
    config: SmartGenAnomalyRunConfig,
    train_pkl: Path,
    out_dir: Path,
) -> dict[str, Any]:
    if config.weight_prior_json is None:
        return {
            "weight_prior_json": "",
            "train_weights_pkl": "",
            "train_weight_scores_path": "",
            "weight_top_k_edges": "",
            "weight_min_edge_weight": "",
            "weight_floor": "",
            "weight_power": "",
            "train_weight_min": "",
            "train_weight_mean": "",
            "train_weight_max": "",
        }
    train_sequences = load_pickle(train_pkl)
    weights, scores = _causal_training_weights(
        train_sequences,
        prior_json=config.weight_prior_json,
        top_k_edges=config.weight_top_k_edges,
        min_edge_weight=config.weight_min_edge_weight,
        weight_floor=config.weight_floor,
        weight_power=config.weight_power,
    )
    weights_path = out_dir / f"{config.tag}_train_weights.pkl"
    scores_path = out_dir / f"{config.tag}_train_weight_scores.json"
    save_pickle(weights_path, weights)
    scores_path.write_text(json.dumps(_jsonable(scores), ensure_ascii=False, indent=2), encoding="utf-8")
    arr = np.asarray(weights, dtype=np.float32)
    return {
        "weight_prior_json": str(config.weight_prior_json.resolve()),
        "train_weights_pkl": str(weights_path),
        "train_weight_scores_path": str(scores_path),
        "weight_top_k_edges": config.weight_top_k_edges,
        "weight_min_edge_weight": config.weight_min_edge_weight if config.weight_min_edge_weight is not None else "",
        "weight_floor": config.weight_floor,
        "weight_power": config.weight_power,
        "train_weight_min": float(np.min(arr)) if len(arr) else math.nan,
        "train_weight_mean": float(np.mean(arr)) if len(arr) else math.nan,
        "train_weight_max": float(np.max(arr)) if len(arr) else math.nan,
    }


def run_smartgen_anomaly_experiment(config: SmartGenAnomalyRunConfig) -> dict[str, Any]:
    if config.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

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
    weight_payload = _prepare_training_weights(config, train_pkl, out_dir)

    threshold_vld_pkl = (config.validation_pkl or vld_pkl).resolve()
    threshold_vld_size = len(load_pickle(threshold_vld_pkl)) if threshold_vld_pkl.exists() else None

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
        "requested_device": config.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dry_run": config.dry_run,
        "synthetic_pkl": str(config.synthetic_pkl.resolve()),
        "synthetic_size": len(synthetic_data),
        "train_pkl": str(train_pkl),
        "vld_pkl": str(vld_pkl),
        "threshold_vld_pkl": str(threshold_vld_pkl),
        "train_size": train_size,
        "vld_size": vld_size,
        "threshold_vld_size": threshold_vld_size,
        "model_path": str(model_path),
        "result_path": str(result_path),
        "attack_pkl": str((config.attack_pkl or defaults["attack_pkl"]).resolve()),
        "target_test_pkl": str((config.target_test_pkl or defaults["target_test_pkl"]).resolve()),
    }
    payload.update(weight_payload)
    if config.dry_run:
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    device = _resolve_torch_device(config.device)
    models_module = _import_smartgen_models(smartgen_root)
    vocab_size = VOCAB_DIC[config.dataset]
    seq_len = 10
    _setup_torch_seed(config.seed)
    train_history = _train_adaptive(
        models_module,
        config.env,
        vocab_size,
        config.epochs,
        str(train_pkl),
        str(model_path),
        seq_len,
        device,
        train_weights_file=weight_payload["train_weights_pkl"] or None,
    )
    learned_threshold, validation_losses = _find_threshold_adaptive(
        models_module,
        config.env,
        vocab_size,
        str(threshold_vld_pkl),
        str(model_path),
        seq_len,
        threshold_percentage,
        device,
    )
    metrics = _evaluate_adaptive(
        models_module,
        config.env,
        vocab_size,
        str(config.attack_pkl or defaults["attack_pkl"]),
        str(config.target_test_pkl or defaults["target_test_pkl"]),
        str(model_path),
        seq_len,
        threshold=learned_threshold,
        device=device,
    )
    payload.update(
        {
            "device": str(device),
            "train_history": train_history,
            "validation_loss_avg": float(np.mean(validation_losses)) if validation_losses else math.nan,
            "validation_loss_min": min(validation_losses) if validation_losses else math.nan,
            "validation_loss_max": max(validation_losses) if validation_losses else math.nan,
            "learned_threshold": float(learned_threshold),
        }
    )
    payload.update(metrics)
    result_path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


SMARTGEN_SWEEP_FIELDS = [
    "tag",
    "slug",
    "synthetic_pkl",
    "synthetic_size",
    "train_size",
    "vld_size",
    "threshold_vld_pkl",
    "threshold_vld_size",
    "recall",
    "precision",
    "accuracy",
    "F1 score",
    "learned_threshold",
    "device",
    "weight_prior_json",
    "weight_floor",
    "weight_power",
    "train_weight_mean",
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
        "threshold_vld_pkl": payload.get("threshold_vld_pkl", ""),
        "threshold_vld_size": payload.get("threshold_vld_size", ""),
        "recall": payload.get("recall", ""),
        "precision": payload.get("precision", ""),
        "accuracy": payload.get("accuracy", ""),
        "F1 score": payload.get("F1 score", ""),
        "learned_threshold": payload.get("learned_threshold", ""),
        "device": payload.get("device", ""),
        "weight_prior_json": payload.get("weight_prior_json", ""),
        "weight_floor": payload.get("weight_floor", ""),
        "weight_power": payload.get("weight_power", ""),
        "train_weight_mean": payload.get("train_weight_mean", ""),
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
            device=base_config.device,
            cuda_visible_devices=base_config.cuda_visible_devices,
            dry_run=base_config.dry_run,
            attack_pkl=base_config.attack_pkl,
            target_test_pkl=base_config.target_test_pkl,
            validation_pkl=base_config.validation_pkl,
            weight_prior_json=base_config.weight_prior_json,
            weight_top_k_edges=base_config.weight_top_k_edges,
            weight_min_edge_weight=base_config.weight_min_edge_weight,
            weight_floor=base_config.weight_floor,
            weight_power=base_config.weight_power,
        )
        payload = run_smartgen_anomaly_experiment(config)
        summary_rows.append(summarize_smartgen_payload(payload, slug=slug))
    write_smartgen_sweep_summary(summary_rows, base_config.out_dir)
    return summary_rows
