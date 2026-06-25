#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import importlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.schema import load_numeric_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and pack GPT-5.5 Gen flat-quadruple JSONL.")
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--out-pkl", required=True, type=Path)
    parser.add_argument("--out-validation-report", required=True, type=Path)
    parser.add_argument("--out-generation-report", required=True, type=Path)
    parser.add_argument("--dictionary-py", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--scenario-key", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--expected-count", type=int, default=50)
    parser.add_argument("--expected-length", type=int)
    parser.add_argument("--source-pkl", type=Path)
    parser.add_argument("--target-pkl", type=Path)
    parser.add_argument("--schema-json", type=Path)
    parser.add_argument("--guard-report-json", type=Path)
    parser.add_argument("--guarded-hints-json", type=Path)
    parser.add_argument("--resolved-causal-relation-prior-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input_jsonl)
    vocab = load_vocab(args.dictionary_py, args.dataset)

    invalid: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    seen: Counter[tuple[int, ...]] = Counter()
    lengths: Counter[int] = Counter()

    for index, row in enumerate(rows):
        sequence = row.get("sequence") if isinstance(row, Mapping) else None
        sequence_id = row.get("sequence_id", str(index)) if isinstance(row, Mapping) else str(index)
        flat, extraction_error = extract_flat(sequence)
        reasons: list[str] = []
        if extraction_error:
            reasons.append(extraction_error)
        if flat is not None:
            reasons.extend(validate_flat(flat, vocab, expected_length=args.expected_length))
        if reasons:
            invalid.append({"index": index, "sequence_id": sequence_id, "reasons": reasons, "preview": flat[:16] if flat else None})
            continue
        assert flat is not None
        clean_rows.append({"sequence_id": str(sequence_id), "sequence": flat, "notes": row.get("notes")})
        seen[tuple(flat)] += 1
        lengths[len(flat)] += 1

    duplicate_sequences = sum(count - 1 for count in seen.values() if count > 1)
    status = "valid" if not invalid and len(clean_rows) == args.expected_count else "invalid"
    validation_report = {
        "status": status,
        "input_jsonl": str(args.input_jsonl.resolve()),
        "out_pkl": str(args.out_pkl.resolve()),
        "num_rows": len(rows),
        "num_valid": len(clean_rows),
        "num_invalid": len(invalid),
        "expected_count": args.expected_count,
        "expected_length": args.expected_length,
        "length_distribution": {str(length): count for length, count in sorted(lengths.items())},
        "unique_sequence_count": len(seen),
        "duplicate_sequence_count": duplicate_sequences,
        "duplicate_rate": duplicate_sequences / len(clean_rows) if clean_rows else None,
        "field_order": ["day", "hour_slot", "device_id", "action_id"],
        "field_range_checks": {"day": "0..6", "hour_slot": "0..7", "device_id": "dictionary", "action_id": "dictionary"},
        "invalid_sequences": invalid,
    }
    if status != "valid":
        write_json(args.out_validation_report, validation_report)
        raise ValueError(f"GPT-5.5 generation validation failed: {len(invalid)} invalid rows, {len(clean_rows)} valid rows")

    sequences = load_numeric_sequences([row["sequence"] for row in clean_rows])
    args.out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_pkl, "wb") as f:
        pickle.dump([seq.to_flat_numeric() for seq in sequences], f)

    generation_report = {
        "generator": "gpt55_generation",
        "generation_model": "GPT-5.5",
        "api_llm": False,
        "manual_generation": True,
        "gpt55_generation_assisted": True,
        "num_generated": len(clean_rows),
        "dataset": args.dataset,
        "scenario": args.scenario,
        "scenario_key": args.scenario_key,
        "seed": args.seed,
        "sequence_format": "flat_quadruples",
        "sequence_length": args.expected_length,
        "jsonl": str(args.input_jsonl.resolve()),
        "pkl": str(args.out_pkl.resolve()),
        "schema_json": resolved_or_none(args.schema_json),
        "guard_report_json": resolved_or_none(args.guard_report_json),
        "guarded_hints_json": resolved_or_none(args.guarded_hints_json),
        "resolved_causal_relation_prior_json": resolved_or_none(args.resolved_causal_relation_prior_json),
        "source_pkl": resolved_or_none(args.source_pkl),
        "target_pkl": resolved_or_none(args.target_pkl),
        "notes": (
            "Sequence content was authored by the GPT-5.5-assisted generation process from the prompt package, "
            "guarded GSS hints, causal relation prior, dictionary, and source/target distribution inspection. "
            "This script only validated and packed the authored JSONL."
        ),
    }
    write_json(args.out_validation_report, validation_report)
    write_json(args.out_generation_report, generation_report)
    print(f"validated GPT-5.5 rows: {len(clean_rows)}")
    print(f"saved pkl: {args.out_pkl.resolve()}")
    print(f"saved validation report: {args.out_validation_report.resolve()}")
    print(f"saved generation report: {args.out_generation_report.resolve()}")


def validate_flat(flat: Sequence[int], vocab: Mapping[str, Any], expected_length: int | None = None) -> list[str]:
    reasons: list[str] = []
    if not flat:
        return ["empty_sequence"]
    if len(flat) % 4 != 0:
        reasons.append("length_not_multiple_of_4")
    if expected_length is not None and len(flat) != expected_length:
        reasons.append(f"length_mismatch:{len(flat)}!=expected_{expected_length}")
    legal_devices = vocab["legal_devices"]
    legal_actions = vocab["legal_actions"]
    action_to_device = vocab["action_to_device"]
    for offset in range(0, len(flat) - len(flat) % 4, 4):
        day, hour, device, action = [int(value) for value in flat[offset : offset + 4]]
        event_index = offset // 4
        if not 0 <= day <= 6:
            reasons.append(f"day_out_of_range@event_{event_index}:{day}")
        if not 0 <= hour <= 7:
            reasons.append(f"hour_slot_out_of_range@event_{event_index}:{hour}")
        if device not in legal_devices:
            reasons.append(f"illegal_device_id@event_{event_index}:{device}")
        if action not in legal_actions:
            reasons.append(f"illegal_action_id@event_{event_index}:{action}")
        expected_device = action_to_device.get(action)
        if expected_device is not None and device in legal_devices and expected_device != device:
            reasons.append(f"device_action_mismatch@event_{event_index}:device={device},action={action},expected_device={expected_device}")
    return reasons


def load_vocab(path: Path, dataset: str) -> dict[str, Any]:
    payload = load_dictionaries(path)
    prefix = dataset
    device_name_to_id = {str(k): int(v) for k, v in payload[f"{prefix}_devices_dict"].items()}
    action_name_to_id = {str(k): int(v) for k, v in payload[f"{prefix}_actions"].items()}
    action_to_device: dict[int, int] = {}
    action_by_device: dict[int, set[int]] = {}
    for action_name, action_id in action_name_to_id.items():
        device_name = action_name.split(":", 1)[0]
        if device_name not in device_name_to_id:
            continue
        device_id = int(device_name_to_id[device_name])
        action_to_device[int(action_id)] = device_id
        action_by_device.setdefault(device_id, set()).add(int(action_id))
    return {
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
    return rows


def extract_flat(value: Any) -> tuple[list[int] | None, str | None]:
    if value is None:
        return None, "missing_sequence"
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None, f"sequence_not_list:{type(value).__name__}"
    try:
        return [int(item) for item in value], None
    except Exception as exc:
        return None, f"non_integer_value:{exc}"


def resolved_or_none(path: Path | None) -> str | None:
    return str(path.resolve()) if path is not None else None


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def install_numpy_pickle_compat() -> None:
    if "numpy._core" not in sys.modules:
        try:
            sys.modules["numpy._core"] = importlib.import_module("numpy.core")
        except Exception:
            pass


if __name__ == "__main__":
    install_numpy_pickle_compat()
    main()
