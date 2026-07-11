"""运行 Gen 内置的下游异常检测协议。

本模块保持最终评估与 Gen 一致：导入项目内 Gen Transformer autoencoder 和数据集
类，在生成的 normal 数据上训练，用 validation reconstruction loss 的分位数
确定异常阈值，再在 Gen 的 target-normal + attack test set 上评估。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import importlib.util
import json
import math
import numpy as np
import os
import pickle
import random
import sys

from .json_utils import jsonable


VOCAB_DIC = {"fr": 223, "sp": 235, "us": 269}

DATASETS = tuple(VOCAB_DIC)
ENVIRONMENTS = ("spring", "night", "multiple")
ENV_BY_SCENARIO = {
    "st": "spring",
    "tt": "night",
    "nt": "multiple",
    "spring": "spring",
    "night": "night",
    "multiple": "multiple",
}
SCENARIO_BY_ENV = {"spring": "st", "night": "tt", "multiple": "nt"}
SOURCE_ENV_BY_TARGET_ENV = {"spring": "winter", "night": "daytime", "multiple": "single"}
ATTACK_BY_ENV = {
    "spring": "spring_attack_heater",
    "night": "night_attack_time",
    "multiple": "multiple_attack_tv",
}

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


def env_for_scenario(scenario: str) -> str:
    """把 ``st`` 等短别名规范化为 Gen 使用的环境名。"""
    try:
        return ENV_BY_SCENARIO[scenario]
    except KeyError as exc:
        valid = ", ".join(sorted(ENV_BY_SCENARIO))
        raise ValueError(f"scenario must be one of: {valid}") from exc


@dataclass(frozen=True)
class GenDownstreamADRunConfig:
    """一次 Gen downstream AD 运行所需的全部输入和超参数。"""

    gen_root: Path
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


def default_gen_paths(gen_root: str | Path, dataset: str, env: str) -> dict[str, Path]:
    """返回某个 dataset/env 单元格对应的项目内 Gen attack/test 路径。"""
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of: {', '.join(DATASETS)}")
    if env not in ENVIRONMENTS:
        raise ValueError(f"env must be one of: {', '.join(ENVIRONMENTS)}")
    pipeline = Path(gen_root).resolve() / "anomaly_detection_pipeline"
    attack_name = f"labeled_{dataset}_{ATTACK_BY_ENV[env]}.pkl"
    return {
        "pipeline_root": pipeline,
        "attack_pkl": pipeline / "attack" / dataset / attack_name,
        "target_test_pkl": pipeline / "test" / dataset / env / "test.pkl",
    }


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: str | Path, obj: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(obj, f)


def split_generated_to_train_validation(
    data_file: str | Path,
    train_file: str | Path,
    vld_file: str | Path,
    split_ratio: float = 0.8,
    seed: int = 2024,
) -> tuple[list[Any], list[Any]]:
    """按固定 seed 把生成 normal 数据切分为 train/validation。"""
    if not 0.0 < split_ratio < 1.0:
        raise ValueError("split_ratio must be between 0 and 1 (exclusive)")
    data = list(load_pickle(data_file))
    if len(data) < 2:
        raise ValueError("generated data must contain at least 2 rows for train/validation splitting")
    rng = random.Random(seed)
    rng.shuffle(data)
    split_index = int(len(data) * split_ratio)
    train = data[:split_index]
    vld = data[split_index:]
    save_pickle(train_file, train)
    save_pickle(vld_file, vld)
    return train, vld


def validate_run_config(config: GenDownstreamADRunConfig) -> None:
    """在创建输出和启动 GPU 训练前校验实验参数。"""
    if config.dataset not in DATASETS:
        raise ValueError(f"dataset must be one of: {', '.join(DATASETS)}")
    if config.env not in ENVIRONMENTS:
        raise ValueError(f"env must be one of: {', '.join(ENVIRONMENTS)}")
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if not 0.0 < config.split_ratio < 1.0:
        raise ValueError("split_ratio must be between 0 and 1 (exclusive)")
    if config.threshold_percentage is not None and not 0.0 <= config.threshold_percentage <= 100.0:
        raise ValueError("threshold_percentage must be between 0 and 100")


def _import_gen_models(gen_root: Path):
    """在不安装 Gen 包的情况下导入 Gen 的 ``models1.py``。"""
    pipeline_root = gen_root / "anomaly_detection_pipeline"
    module_path = pipeline_root / "models1.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Gen models module not found: {module_path}")
    root_str = str(pipeline_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    spec = importlib.util.spec_from_file_location("_csh_gen_models", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import Gen models from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_torch_device(requested: str):
    """解析并校验训练设备，尤其防止请求 CUDA 但实际不可用。"""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Gen anomaly evaluation requires torch for real training runs") from exc
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
    """把扁平序列 pad/truncate 到 Gen 数据集类期望的固定长度。"""
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
    """根据环境选择 Gen 原始代码中的环境专用 Dataset 类。"""
    if env == "spring":
        return models_module.TimeSeriesDataset2
    if env == "night":
        return models_module.TimeSeriesDataset3
    if env == "multiple":
        return models_module.TimeSeriesDataset4
    raise ValueError(f"env must be one of: {', '.join(ENVIRONMENTS)}")


def _make_loader(models_module, env: str, vocab_size: int, data_file: str | Path, batch_size: int):
    from torch.utils.data import DataLoader

    data = np.array(_pad_sequences(vocab_size, load_pickle(data_file)))
    dataset = _dataset_for_env(models_module, env)(vocab_size, data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _sequence_loss(output, src, mask_v, vocab_size: int, seq_len: int, criterion):
    """计算一个 batch 的 masked token reconstruction loss。"""
    import torch

    loss = criterion(output.view(-1, vocab_size), src.view(-1))
    loss = loss.reshape(-1, seq_len) * mask_v
    denom = torch.sum(mask_v)
    if denom.item() == 0:
        return torch.sum(loss)
    return torch.sum(loss) / denom


def _train_adaptive(
    models_module,
    env: str,
    vocab_size: int,
    epochs: int,
    train_file: str | Path,
    model_path: str | Path,
    seq_len: int,
    device,
) -> list[dict[str, float]]:
    """在生成 normal 序列上训练 Gen Transformer autoencoder。"""
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
    train_loader = _make_loader(models_module, env, vocab_size, train_file, batch_size=32)

    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            src, padding_mask, mask_v = batch
            src = src.to(device).long()
            mask_v = mask_v.to(device)
            padding_mask = padding_mask.to(device)
            output = model(src, src_key_padding_mask=padding_mask)
            loss = _sequence_loss(output, src, mask_v, vocab_size, seq_len, criterion)
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
    """根据 validation reconstruction losses 的分位数设置异常阈值。"""
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
    """在 target normal 行和 Gen attack 行上用学习到的阈值评估异常检测。"""
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


def _prepare_split(config: GenDownstreamADRunConfig, train_pkl: Path, vld_pkl: Path) -> tuple[int, int]:
    train, vld = split_generated_to_train_validation(
        config.synthetic_pkl,
        train_pkl,
        vld_pkl,
        split_ratio=config.split_ratio,
        seed=config.seed,
    )
    return len(train), len(vld)


def _prepare_train_validation_files(
    config: GenDownstreamADRunConfig,
    train_pkl: Path,
    vld_pkl: Path,
) -> tuple[Path, Path, Path, int, int, int, str]:
    if config.env == "multiple":
        # SmartGen 原始下游异常检测对 multiple 场景使用完整的过滤后合成数据集
        # 同时进行训练和阈值校准；spring/night 则沿用下面的 80/20 生成数据
        # 切分路径。这里保留原协议，避免评估口径漂移。
        train_path = config.synthetic_pkl.resolve()
        threshold_vld = (config.validation_pkl or config.synthetic_pkl).resolve()
        size = len(load_pickle(train_path))
        threshold_vld_size = len(load_pickle(threshold_vld)) if threshold_vld.exists() else 0
        return (
            train_path,
            threshold_vld,
            threshold_vld,
            size,
            threshold_vld_size,
            threshold_vld_size,
            "smartgen_multiple_full_synthetic_train_and_validation",
        )

    train_size, vld_size = _prepare_split(config, train_pkl, vld_pkl)
    threshold_vld = (config.validation_pkl or vld_pkl).resolve()
    threshold_vld_size = len(load_pickle(threshold_vld)) if threshold_vld.exists() else vld_size
    return train_pkl, vld_pkl, threshold_vld, train_size, vld_size, threshold_vld_size, "generated_split_train_validation"


def run_gen_downstream_ad_experiment(config: GenDownstreamADRunConfig) -> dict[str, Any]:
    """运行完整 Gen downstream AD 协议，并写出 raw metrics payload。"""
    validate_run_config(config)
    if config.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

    gen_root = config.gen_root.resolve()
    defaults = default_gen_paths(gen_root, config.dataset, config.env)
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
    result_path = out_dir / f"{config.tag}_gen_downstream_ad_eval.json"

    synthetic_data = load_pickle(config.synthetic_pkl)
    (
        train_pkl_for_run,
        vld_pkl_for_payload,
        threshold_vld_pkl,
        train_size,
        vld_size,
        threshold_vld_size,
        training_protocol,
    ) = _prepare_train_validation_files(
        config,
        train_pkl,
        vld_pkl,
    )

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
        "training_protocol": training_protocol,
        "synthetic_pkl": str(config.synthetic_pkl.resolve()),
        "synthetic_size": len(synthetic_data),
        "train_pkl": str(train_pkl_for_run),
        "vld_pkl": str(vld_pkl_for_payload),
        "threshold_vld_pkl": str(threshold_vld_pkl),
        "train_size": train_size,
        "vld_size": vld_size,
        "threshold_vld_size": threshold_vld_size,
        "model_path": str(model_path),
        "result_path": str(result_path),
        "attack_pkl": str((config.attack_pkl or defaults["attack_pkl"]).resolve()),
        "target_test_pkl": str((config.target_test_pkl or defaults["target_test_pkl"]).resolve()),
    }
    if config.dry_run:
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    device = _resolve_torch_device(config.device)
    models_module = _import_gen_models(gen_root)
    vocab_size = VOCAB_DIC[config.dataset]
    seq_len = 10
    _setup_torch_seed(config.seed)
    train_history = _train_adaptive(
        models_module,
        config.env,
        vocab_size,
        config.epochs,
        str(train_pkl_for_run),
        str(model_path),
        seq_len,
        device,
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
    result_path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
