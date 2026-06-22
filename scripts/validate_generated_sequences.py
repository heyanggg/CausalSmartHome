#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import importlib
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.causal_tof import extract_guarded_edges, load_pickle_sequences, save_pickle_sequences, score_sequences_causal_tof
from causal_smart_home.target_distribution_guard import compute_device_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Codex/GPT-5.5 surrogate generated SmartGen sequences.")
    parser.add_argument("--generated-pkl", required=True)
    parser.add_argument("--target-pkl", required=True)
    parser.add_argument("--guard-report-json", required=True)
    parser.add_argument("--guarded-hints-json", required=True)
    parser.add_argument("--dictionary-py", required=True)
    parser.add_argument("--scenario", required=True, choices=["fr_st", "sp_st"])
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--out-invalid-report", required=True)
    parser.add_argument("--out-clean-pkl", required=True)
    parser.add_argument("--max-device-js-warning", type=float, default=0.75)
    parser.add_argument("--max-action-js-warning", type=float, default=0.85)
    parser.add_argument("--max-sp-television-overuse-ratio", type=float, default=1.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [
        Path(args.generated_pkl),
        Path(args.target_pkl),
        Path(args.guard_report_json),
        Path(args.guarded_hints_json),
        Path(args.dictionary_py),
    ]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"required input not found: {path.resolve()}")

    raw = load_pickle_raw(Path(args.generated_pkl))
    target = load_pickle_sequences(args.target_pkl)
    guard_report = read_json(Path(args.guard_report_json))
    hints_payload = read_json(Path(args.guarded_hints_json))
    guarded_edges = extract_guarded_edges(hints_payload)
    vocab = load_vocab(Path(args.dictionary_py), args.scenario)

    invalid_rows: list[dict[str, Any]] = []
    clean_flats: list[list[int]] = []
    for index, item in enumerate(raw):
        flat, extraction_error = extract_flat(item)
        reasons = []
        if extraction_error:
            reasons.append(extraction_error)
        if flat is None:
            invalid_rows.append({"index": index, "reasons": reasons or ["not_a_sequence"], "length": None})
            continue
        reasons.extend(validate_flat(flat, vocab))
        if reasons:
            invalid_rows.append({"index": index, "reasons": reasons, "length": len(flat), "preview": flat[:16]})
            continue
        clean_flats.append(flat)

    clean_sequences = load_pickle_sequences_from_flats(clean_flats)
    target_distribution = compute_device_distribution(target)
    scores = score_sequences_causal_tof(clean_sequences, guarded_edges, target_distribution=target_distribution, mode="weight") if clean_sequences else []
    generated_distribution = compute_device_distribution(clean_sequences)
    television = television_stats(generated_distribution, target_distribution, vocab, args.scenario)
    distribution = distribution_stats(clean_sequences, target)
    causal = causal_stats(scores, guarded_edges)
    warnings = collect_warnings(distribution, television, args)

    report = {
        "status": "valid" if not invalid_rows else "cleaned",
        "generator_expected": "codex_gpt55_surrogate",
        "api_llm": False,
        "surrogate_for_smartgen_llm": True,
        "scenario": args.scenario,
        "generated_pkl": str(Path(args.generated_pkl).resolve()),
        "target_pkl": str(Path(args.target_pkl).resolve()),
        "raw_sequence_count": len(raw),
        "clean_sequence_count": len(clean_flats),
        "invalid_sequence_count": len(invalid_rows),
        "empty_sequence_count": sum(1 for row in invalid_rows if "empty_sequence" in row["reasons"]),
        "non_multiple_of_4_count": sum(1 for row in invalid_rows if "length_not_multiple_of_4" in row["reasons"]),
        "illegal_device_or_action_count": sum(
            1
            for row in invalid_rows
            if any(reason.startswith("illegal_device_id") or reason.startswith("illegal_action_id") for reason in row["reasons"])
        ),
        "device_action_mismatch_count": sum(1 for row in invalid_rows if any(reason.startswith("device_action_mismatch") for reason in row["reasons"])),
        "field_order": ["day", "hour_slot", "device_id", "action_id"],
        "field_range_checks": {"day": "0..6", "hour_slot": "0..7", "device_id": "dictionary", "action_id": "dictionary"},
        "distribution": distribution,
        "television": television,
        "causal": causal,
        "warnings": warnings,
        "guard_report_json": str(Path(args.guard_report_json).resolve()),
        "guarded_hints_json": str(Path(args.guarded_hints_json).resolve()),
        "clean_pkl": str(Path(args.out_clean_pkl).resolve()),
    }
    invalid_report = {
        "generated_pkl": str(Path(args.generated_pkl).resolve()),
        "invalid_sequence_count": len(invalid_rows),
        "invalid_sequences": invalid_rows,
    }
    write_json(Path(args.out_report), report)
    write_json(Path(args.out_invalid_report), invalid_report)
    save_pickle_sequences(args.out_clean_pkl, clean_sequences)
    print(f"validated raw sequences: {len(raw)}")
    print(f"clean sequences: {len(clean_flats)}")
    print(f"invalid sequences: {len(invalid_rows)}")
    print(f"saved validation report: {Path(args.out_report).resolve()}")
    print(f"saved clean pkl: {Path(args.out_clean_pkl).resolve()}")


def validate_flat(flat: Sequence[int], vocab: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if len(flat) == 0:
        return ["empty_sequence"]
    if len(flat) % 4 != 0:
        return ["length_not_multiple_of_4"]
    legal_devices = vocab["legal_devices"]
    legal_actions = vocab["legal_actions"]
    action_to_device = vocab["action_to_device"]
    for i in range(0, len(flat), 4):
        day, hour, device, action = [int(value) for value in flat[i : i + 4]]
        if not 0 <= day <= 6:
            reasons.append(f"day_out_of_range@event_{i//4}:{day}")
        if not 0 <= hour <= 7:
            reasons.append(f"hour_slot_out_of_range@event_{i//4}:{hour}")
        if device not in legal_devices:
            reasons.append(f"illegal_device_id@event_{i//4}:{device}")
        if action not in legal_actions:
            reasons.append(f"illegal_action_id@event_{i//4}:{action}")
        expected_device = action_to_device.get(action)
        if expected_device is not None and device in legal_devices and expected_device != device:
            reasons.append(f"device_action_mismatch@event_{i//4}:device={device},action={action},expected_device={expected_device}")
    return reasons


def distribution_stats(generated: Sequence, target: Sequence) -> dict[str, Any]:
    return {
        "device_js_to_target": js_for_level(generated, target, "device") if generated else None,
        "action_js_to_target": js_for_level(generated, target, "action") if generated else None,
        "transition_js_to_target": transition_js(generated, target) if generated else None,
        "generated_device_distribution": compute_device_distribution(generated),
        "target_device_distribution": compute_device_distribution(target),
    }


def causal_stats(scores: Sequence[Mapping[str, Any]], guarded_edges: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    nonzero_edges = [
        edge
        for edge in guarded_edges
        if float(edge.get("guarded_causal_strength", edge.get("guarded_weight", edge.get("weight", 0.0))) or 0.0) > 0.0
    ]
    return {
        "nonzero_guarded_edges": len(nonzero_edges),
        "mean_causal_coverage": mean([float(score.get("causal_coverage", 0.0)) for score in scores]),
        "mean_causal_violation_rate": mean([float(score.get("causal_violation", 0.0)) for score in scores]),
        "low_evidence_rate": mean([1.0 if float(score.get("checked_edge_weight", 0.0)) == 0.0 else 0.0 for score in scores]),
        "mean_checked_edge_weight": mean([float(score.get("checked_edge_weight", 0.0)) for score in scores]),
        "mean_missing_edge_weight": mean([float(score.get("missing_edge_weight", 0.0)) for score in scores]),
    }


def television_stats(generated_dist: Mapping[str, float], target_dist: Mapping[str, float], vocab: Mapping[str, Any], scenario: str) -> dict[str, Any]:
    tv_id = vocab.get("device_name_to_id", {}).get("Television")
    if tv_id is None:
        return {
            "television_device_key": None,
            "generated_frequency": None,
            "target_frequency": None,
            "overuse_ratio": None,
        }
    tv_key = f"d:{int(tv_id)}"
    generated_freq = float(generated_dist.get(tv_key, 0.0))
    target_freq = float(target_dist.get(tv_key, 0.0))
    ratio = generated_freq / max(target_freq, 1e-8)
    return {
        "scenario": scenario,
        "television_device_key": tv_key,
        "generated_frequency": generated_freq,
        "target_frequency": target_freq,
        "overuse_ratio": ratio,
    }


def collect_warnings(distribution: Mapping[str, Any], television: Mapping[str, Any], args: argparse.Namespace) -> list[str]:
    warnings: list[str] = []
    device_js = distribution.get("device_js_to_target")
    action_js = distribution.get("action_js_to_target")
    if device_js is not None and float(device_js) > args.max_device_js_warning:
        warnings.append(f"device_js_to_target_high:{float(device_js):.6f}")
    if action_js is not None and float(action_js) > args.max_action_js_warning:
        warnings.append(f"action_js_to_target_high:{float(action_js):.6f}")
    if args.scenario == "sp_st" and television.get("overuse_ratio") is not None:
        if float(television["overuse_ratio"]) > args.max_sp_television_overuse_ratio:
            warnings.append(f"sp_television_overuse_ratio_high:{float(television['overuse_ratio']):.6f}")
    return warnings


def load_vocab(path: Path, scenario: str) -> dict[str, Any]:
    payload = load_dictionaries(path)
    prefix = "fr" if scenario == "fr_st" else "sp"
    device_name_to_id = {str(k): int(v) for k, v in payload[f"{prefix}_devices_dict"].items()}
    action_name_to_id = {str(k): int(v) for k, v in payload[f"{prefix}_actions"].items()}
    action_to_device: dict[int, int] = {}
    action_by_device: defaultdict[int, set[int]] = defaultdict(set)
    for action_name, action_id in action_name_to_id.items():
        device_name = action_name.split(":", 1)[0]
        if device_name not in device_name_to_id:
            continue
        device_id = device_name_to_id[device_name]
        action_to_device[int(action_id)] = int(device_id)
        action_by_device[int(device_id)].add(int(action_id))
    return {
        "device_name_to_id": device_name_to_id,
        "action_name_to_id": action_name_to_id,
        "legal_devices": set(device_name_to_id.values()),
        "legal_actions": set(action_name_to_id.values()),
        "action_to_device": action_to_device,
        "action_by_device": {key: sorted(value) for key, value in action_by_device.items()},
    }


def load_dictionaries(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except Exception:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                out[target.id] = value
    return out


def load_pickle_raw(path: Path) -> list[Any]:
    install_numpy_pickle_compat()
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return list(raw)


def extract_flat(item: Any) -> tuple[list[int] | None, str | None]:
    if hasattr(item, "tolist"):
        item = item.tolist()
    if isinstance(item, tuple):
        for part in item:
            flat, error = extract_flat(part)
            if flat is not None:
                return flat, None
        return None, "tuple_without_sequence_payload"
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
        try:
            return [int(value) for value in item], None
        except Exception as exc:
            return None, f"non_integer_value:{exc}"
    return None, f"unsupported_item_type:{type(item).__name__}"


def install_numpy_pickle_compat() -> None:
    if "numpy._core" not in sys.modules:
        try:
            sys.modules["numpy._core"] = importlib.import_module("numpy.core")
        except Exception:
            pass
    for submodule in ("multiarray", "numeric", "umath"):
        old_name = f"numpy.core.{submodule}"
        new_name = f"numpy._core.{submodule}"
        if new_name not in sys.modules:
            try:
                sys.modules[new_name] = importlib.import_module(old_name)
            except Exception:
                pass


def load_pickle_sequences_from_flats(flats: Sequence[Sequence[int]]):
    from causal_smart_home.schema import load_numeric_sequences

    return load_numeric_sequences(flats)


def js_for_level(a: Sequence, b: Sequence, level: str) -> float:
    return jensen_shannon(distribution_for_level(a, level), distribution_for_level(b, level))


def transition_js(a: Sequence, b: Sequence) -> float:
    return jensen_shannon(transition_distribution(a), transition_distribution(b))


def distribution_for_level(sequences: Sequence, level: str) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for seq in sequences:
        for event in seq:
            counts[event.key(level)] += 1
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()} if total else {}


def transition_distribution(sequences: Sequence) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for seq in sequences:
        keys = [event.key("device") for event in seq]
        for src, tgt in zip(keys, keys[1:]):
            counts[f"{src}->{tgt}"] += 1
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()} if total else {}


def jensen_shannon(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    pp = {key: float(p.get(key, 0.0)) for key in keys}
    qq = {key: float(q.get(key, 0.0)) for key in keys}
    mp = {key: 0.5 * (pp[key] + qq[key]) for key in keys}
    return math.sqrt(0.5 * kl(pp, mp) + 0.5 * kl(qq, mp))


def kl(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    total = 0.0
    for key, value in p.items():
        if value > 0 and q.get(key, 0.0) > 0:
            total += value * math.log(value / q[key], 2)
    return total


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
