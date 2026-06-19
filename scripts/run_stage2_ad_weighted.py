#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.smartgen_experiment import (
    DEFAULT_THRESHOLD_PERCENTAGES,
    VOCAB_DIC,
    _dataset_for_env,
    _evaluate_adaptive,
    _find_threshold_adaptive,
    _import_smartgen_models,
    _pad_sequences,
    _resolve_torch_device,
    _sequence_loss_per_sample,
    _setup_torch_seed,
    load_pickle,
    save_pickle,
)


SMARTGEN_ROOT = Path("/home/heyang/projects/SmartGen")
CSH_ROOT = Path("/home/heyang/projects/CausalSmartHome")

DATASETS = {
    "fr_st": {
        "dataset": "fr",
        "env": "spring",
        "tof": SMARTGEN_ROOT / "SmartGen/filter_data/fr/spring/fr_spring_generation_SPPC_th=0.918_gpt-4o_seq_filter_true.pkl",
        "causal_scores": CSH_ROOT / "outputs/gen_gcad/fr_st_device_score/smartgen_tof_causal_scores.json",
        "attack": SMARTGEN_ROOT / "anomaly_detection_pipeline/attack/fr/labeled_fr_spring_attack_heater.pkl",
        "target_test": SMARTGEN_ROOT / "anomaly_detection_pipeline/test/fr/spring/test.pkl",
    },
    "sp_st": {
        "dataset": "sp",
        "env": "spring",
        "tof": SMARTGEN_ROOT / "SmartGen/filter_data/sp/spring/sp_spring_generation_SPPC_th=0.915_gpt-4o_seq_filter_true.pkl",
        "causal_scores": CSH_ROOT / "outputs/gen_gcad/sp_st_device_score/smartgen_tof_causal_scores.json",
        "attack": SMARTGEN_ROOT / "anomaly_detection_pipeline/attack/sp/labeled_sp_spring_attack_heater.pkl",
        "target_test": SMARTGEN_ROOT / "anomaly_detection_pipeline/test/sp/spring/test.pkl",
    },
}

METHODS = ("tof_baseline", "duplicated_soft_v1", "score_weighted", "violation_penalty", "binary_safe")


def split_indices(n: int, split_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split_at = int(n * split_ratio)
    return indices[:split_at], indices[split_at:]


def select_rows(rows: Sequence[Any], indices: Sequence[int]) -> list[Any]:
    return [rows[i] for i in indices]


def clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def load_scores(path: Path, expected_len: int) -> list[dict[str, Any]]:
    scores = json.loads(path.read_text(encoding="utf-8"))
    scores = sorted(scores, key=lambda item: int(item["index"]))
    if len(scores) != expected_len:
        raise ValueError(f"score length {len(scores)} does not match TOF length {expected_len}: {path}")
    for expected, score in enumerate(scores):
        if int(score["index"]) != expected:
            raise ValueError(f"score index mismatch at {expected}: {score['index']}")
    return scores


def build_causal_weights(
    scores: Sequence[dict[str, Any]],
    train_idx: Sequence[int],
    lambda_score: float,
    lambda_violation: float,
) -> list[dict[str, Any]]:
    train_scored = [scores[i] for i in train_idx if not scores[i]["low_evidence"]]
    mean_score = float(np.mean([float(item["causal_score"]) for item in train_scored])) if train_scored else 0.0
    rows: list[dict[str, Any]] = []
    for score in scores:
        causal_score = float(score["causal_score"])
        violation_rate = float(score["violation_rate"])
        low_evidence = bool(score["low_evidence"])
        if low_evidence:
            score_weight = 1.0
            violation_weight = 1.0
        else:
            score_weight = clip(1.0 + lambda_score * (causal_score - mean_score), 0.75, 1.25)
            violation_weight = clip(1.0 - lambda_violation * violation_rate, 0.75, 1.0)
        rows.append(
            {
                "original_index": int(score["index"]),
                "causal_score": causal_score,
                "causal_coverage": float(score["causal_coverage"]),
                "violation_rate": violation_rate,
                "low_evidence": low_evidence,
                "weight_score_weighted": score_weight,
                "weight_violation_penalty": violation_weight,
                "weight_binary_safe": 0.75 if (violation_rate >= 0.5 and not low_evidence) else 1.0,
                "train_mean_score_non_low_evidence": mean_score,
            }
        )
    return rows


def write_weights(out_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    (out_dir / "causal_weights.json").write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_dir / "causal_weights.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "original_index",
            "causal_score",
            "causal_coverage",
            "violation_rate",
            "low_evidence",
            "weight_score_weighted",
            "weight_violation_penalty",
            "weight_binary_safe",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def duplicated_train_indices(scores: Sequence[dict[str, Any]], train_idx: Sequence[int]) -> list[int]:
    out: list[int] = []
    for idx in train_idx:
        score = scores[idx]
        causal_score = float(score["causal_score"])
        if score["low_evidence"]:
            out.append(idx)
        elif causal_score >= 0.6:
            out.extend([idx, idx])
        elif causal_score >= 0.3:
            out.append(idx)
    return out


def weights_for_method(method: str, weight_rows: Sequence[dict[str, Any]], train_original_indices: Sequence[int]) -> list[float] | None:
    if method in {"tof_baseline", "duplicated_soft_v1"}:
        return None
    field = {
        "score_weighted": "weight_score_weighted",
        "violation_penalty": "weight_violation_penalty",
        "binary_safe": "weight_binary_safe",
    }[method]
    by_index = {int(row["original_index"]): row for row in weight_rows}
    return [float(by_index[idx][field]) for idx in train_original_indices]


def make_loader(models_module, env: str, vocab_size: int, rows: Sequence[Sequence[int]], weights: Sequence[float] | None, batch_size: int):
    from torch.utils.data import DataLoader, Dataset

    data = np.asarray(_pad_sequences(vocab_size, rows))
    dataset = _dataset_for_env(models_module, env)(vocab_size, data)

    if weights is None:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    arr = np.asarray(weights, dtype=np.float32)
    if len(arr) != len(dataset):
        raise ValueError(f"weights length {len(arr)} does not match training rows {len(dataset)}")

    class WeightedDataset(Dataset):
        def __len__(self):
            return len(dataset)

        def __getitem__(self, index):
            src, padding_mask, mask_v = dataset[index]
            return src, padding_mask, mask_v, arr[index]

    return DataLoader(WeightedDataset(), batch_size=batch_size, shuffle=False)


def train_model(
    models_module,
    env: str,
    vocab_size: int,
    train_rows: Sequence[Sequence[int]],
    weights: Sequence[float] | None,
    model_path: Path,
    epochs: int,
    seed: int,
    device,
) -> list[dict[str, float]]:
    import torch
    import torch.nn as nn
    from torch import optim

    _setup_torch_seed(seed)
    model = models_module.TransformerAutoencoder(
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
    ).to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loader = make_loader(models_module, env, vocab_size, train_rows, weights, batch_size=32)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        total = 0.0
        for batch in loader:
            if weights is None:
                src, padding_mask, mask_v = batch
                sample_weights = None
            else:
                src, padding_mask, mask_v, sample_weights = batch
            src = src.to(device).long()
            mask_v = mask_v.to(device)
            padding_mask = padding_mask.to(device)
            output = model(src, src_key_padding_mask=padding_mask)
            per_sample = _sequence_loss_per_sample(output, src, mask_v, vocab_size, 10, criterion)
            if sample_weights is not None:
                sample_weights = sample_weights.to(device).to(per_sample.dtype)
                loss = torch.mean(per_sample * sample_weights)
            else:
                loss = torch.mean(per_sample)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
        avg = total / len(loader) if len(loader) else math.nan
        history.append({"epoch": epoch + 1, "train_loss": avg})
        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg:.4f}")
    torch.save(model.state_dict(), model_path)
    return history


def evaluate_method(
    method: str,
    dataset_key: str,
    info: dict[str, Any],
    all_rows: Sequence[Sequence[int]],
    scores: Sequence[dict[str, Any]],
    weight_rows: Sequence[dict[str, Any]],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    out_dir: Path,
    args: argparse.Namespace,
    models_module,
    device,
) -> dict[str, Any]:
    dataset = info["dataset"]
    env = info["env"]
    vocab_size = VOCAB_DIC[dataset]
    method_dir = out_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    val_rows = select_rows(all_rows, val_idx)
    val_pkl = out_dir / "common_tof_val.pkl"
    if not val_pkl.exists():
        save_pickle(val_pkl, val_rows)

    if method == "duplicated_soft_v1":
        train_original_indices = duplicated_train_indices(scores, train_idx)
    else:
        train_original_indices = list(train_idx)
    train_rows = select_rows(all_rows, train_original_indices)
    weights = weights_for_method(method, weight_rows, train_original_indices)

    train_pkl = method_dir / "train.pkl"
    save_pickle(train_pkl, train_rows)
    if weights is not None:
        save_pickle(method_dir / "train_weights.pkl", list(weights))
    (method_dir / "train_original_indices.json").write_text(json.dumps(list(train_original_indices), indent=2), encoding="utf-8")

    model_path = method_dir / "transformer_autoencoder.pth"
    log_path = method_dir / "train_command.log"
    with open(log_path, "w", encoding="utf-8") as log:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = log
        sys.stderr = log
        try:
            print(json.dumps({"method": method, "dataset_key": dataset_key, "train_size": len(train_rows)}, indent=2))
            history = train_model(models_module, env, vocab_size, train_rows, weights, model_path, args.epochs, args.seed, device)
            threshold, val_losses = _find_threshold_adaptive(
                models_module,
                env,
                vocab_size,
                str(val_pkl),
                str(model_path),
                10,
                args.threshold_percentage,
                device,
            )
            metrics = _evaluate_adaptive(
                models_module,
                env,
                vocab_size,
                str(info["attack"]),
                str(info["target_test"]),
                str(model_path),
                10,
                threshold,
                device,
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    tp = int(metrics.get("TP", 0))
    tn = int(metrics.get("TN", 0))
    fp = int(metrics.get("FP", 0))
    fn = int(metrics.get("FN", 0))
    train_set = set(train_original_indices)
    val_set = set(val_idx)
    train_weight_arr = np.asarray(weights if weights is not None else [1.0] * len(train_rows), dtype=np.float32)
    payload = {
        "dataset_key": dataset_key,
        "method": method,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "F1": metrics["F1 score"],
        "FPR": fp / (fp + tn) if fp + tn else 0.0,
        "FNR": fn / (fn + tp) if fn + tp else 0.0,
        "accuracy": metrics["accuracy"],
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "mean_train_weight": float(np.mean(train_weight_arr)) if len(train_weight_arr) else None,
        "min_train_weight": float(np.min(train_weight_arr)) if len(train_weight_arr) else None,
        "max_train_weight": float(np.max(train_weight_arr)) if len(train_weight_arr) else None,
        "threshold_value": float(threshold),
        "threshold_percentile": args.threshold_percentage,
        "sample_duplication_used": method == "duplicated_soft_v1",
        "common_validation_used": True,
        "duplicate_leakage_check": len(train_set & val_set) == 0,
        "duplicate_leakage_count": len(train_set & val_set),
        "unique_original_train_sample_count": len(train_set),
        "unique_validation_sample_count": len(val_set),
        "train_history": history,
        "validation_loss_avg": float(np.mean(val_losses)) if val_losses else None,
        "train_pkl": str(train_pkl),
        "val_pkl": str(val_pkl),
        "model_path": str(model_path),
        "train_command_log": str(log_path),
    }
    (method_dir / "metrics.json").write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


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


def run_dataset(dataset_key: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    info = DATASETS[dataset_key]
    for key in ("tof", "causal_scores", "attack", "target_test"):
        if not info[key].exists():
            raise FileNotFoundError(f"{dataset_key} missing {key}: {info[key]}")
    out_dir = CSH_ROOT / "outputs/stage2_ad_weighted" / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(load_pickle(info["tof"]))
    train_idx, val_idx = split_indices(len(rows), args.split_ratio, args.seed)
    scores = load_scores(info["causal_scores"], len(rows))
    weight_rows = build_causal_weights(scores, train_idx, args.lambda_score, args.lambda_violation)
    write_weights(out_dir, weight_rows)
    (out_dir / "split_indices.json").write_text(
        json.dumps({"train_idx": train_idx, "val_idx": val_idx}, indent=2),
        encoding="utf-8",
    )
    save_pickle(out_dir / "common_tof_val.pkl", select_rows(rows, val_idx))

    threshold_percentage = (
        args.threshold_percentage
        if args.threshold_percentage is not None
        else DEFAULT_THRESHOLD_PERCENTAGES[(info["dataset"], info["env"])]
    )
    args_for_run = argparse.Namespace(**vars(args))
    args_for_run.threshold_percentage = threshold_percentage

    if args.dry_run:
        return []

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    device = _resolve_torch_device(args.device)
    models_module = _import_smartgen_models(SMARTGEN_ROOT)
    return [
        evaluate_method(method, dataset_key, info, rows, scores, weight_rows, train_idx, val_idx, out_dir, args_for_run, models_module, device)
        for method in METHODS
    ]


def write_summary(all_rows: Sequence[dict[str, Any]]) -> None:
    out_dir = CSH_ROOT / "outputs/stage2_ad_weighted"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    fields = [
        "dataset_key",
        "method",
        "precision",
        "recall",
        "F1",
        "FPR",
        "FNR",
        "train_size",
        "val_size",
        "mean_train_weight",
        "min_train_weight",
        "max_train_weight",
        "threshold_value",
        "threshold_percentile",
        "sample_duplication_used",
        "common_validation_used",
        "duplicate_leakage_check",
        "unique_original_train_sample_count",
        "unique_validation_sample_count",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    (out_dir / "summary.json").write_text(json.dumps(jsonable(list(all_rows)), ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Stage 2.1 Weighted AD Summary",
        "",
        "Common validation: original SmartGen TOF is split once per dataset; every method uses the same TOF validation split for thresholding.",
        "",
        "| dataset | method | precision | recall | F1 | FPR | FNR | train | val | mean w | min w | max w | threshold | dup | leak-free |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in all_rows:
        lines.append(
            f"| {row['dataset_key']} | {row['method']} | {fmt(row['precision'])} | {fmt(row['recall'])} | "
            f"{fmt(row['F1'])} | {fmt(row['FPR'])} | {fmt(row['FNR'])} | {row['train_size']} | {row['val_size']} | "
            f"{fmt(row['mean_train_weight'])} | {fmt(row['min_train_weight'])} | {fmt(row['max_train_weight'])} | "
            f"{fmt(row['threshold_value'])} | {row['sample_duplication_used']} | {row['duplicate_leakage_check']} |"
        )
    lines.extend(["", "## Readout", ""])
    for dataset_key in sorted({row["dataset_key"] for row in all_rows}):
        subset = {row["method"]: row for row in all_rows if row["dataset_key"] == dataset_key}
        baseline = subset.get("tof_baseline")
        duplicated = subset.get("duplicated_soft_v1")
        if baseline and duplicated:
            lines.append(
                f"- {dataset_key}: duplicated soft changes F1 by {duplicated['F1'] - baseline['F1']:+.6f} "
                f"and FPR by {duplicated['FPR'] - baseline['FPR']:+.6f} vs TOF baseline."
            )
        for method in ("score_weighted", "violation_penalty", "binary_safe"):
            row = subset.get(method)
            if baseline and row:
                lines.append(
                    f"- {dataset_key} {method}: recall {row['recall']:.6f}, FPR delta {row['FPR'] - baseline['FPR']:+.6f}, "
                    f"F1 delta {row['F1'] - baseline['F1']:+.6f}."
                )
    lines.append("")
    lines.append(f"CSV: `{csv_path}`")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage 2.1 weighted AD with common TOF validation")
    parser.add_argument("--datasets", default="fr_st,sp_st")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--threshold-percentage", type=float)
    parser.add_argument("--lambda-score", type=float, default=0.5)
    parser.add_argument("--lambda-violation", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [item.strip() for item in args.datasets.split(",") if item.strip()]
    unknown = [item for item in selected if item not in DATASETS]
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    all_rows: list[dict[str, Any]] = []
    for dataset_key in selected:
        all_rows.extend(run_dataset(dataset_key, args))
    if all_rows:
        write_summary(all_rows)
    print(json.dumps({"datasets": selected, "dry_run": args.dry_run, "rows": len(all_rows)}, indent=2))


if __name__ == "__main__":
    main()
