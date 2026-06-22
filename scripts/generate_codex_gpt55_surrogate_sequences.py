#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import importlib
import json
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fresh Codex/GPT-5.5 surrogate SmartGen flat-quadruple sequences.")
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--scenario", required=True, choices=["fr_st", "sp_st"])
    parser.add_argument("--target-pkl", required=True)
    parser.add_argument("--source-pkl", required=True)
    parser.add_argument("--old-generated-pkl")
    parser.add_argument("--dictionary-py", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-pkl", required=True)
    parser.add_argument("--out-metadata", required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--fallback-count", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_dir = Path(args.package_dir).resolve()
    target_pkl = Path(args.target_pkl).resolve()
    source_pkl = Path(args.source_pkl).resolve()
    dictionary_py = Path(args.dictionary_py).resolve()
    for path in (package_dir, target_pkl, source_pkl, dictionary_py):
        if not path.exists():
            raise FileNotFoundError(f"required input not found: {path}")

    schema = read_json(package_dir / "generation_schema.json")
    guard_report = read_json(package_dir / "guard_report.json")
    hints = read_json(package_dir / "guarded_reweighted_gss_hints.json")
    target = load_pickle(target_pkl)
    source = load_pickle(source_pkl)
    old = load_pickle(Path(args.old_generated_pkl)) if args.old_generated_pkl and Path(args.old_generated_pkl).exists() else []
    num_sequences, count_reason = resolve_count(args, old)
    length_plan, sequence_length, length_reason = resolve_lengths(num_sequences, old, args.scenario)

    dictionaries = load_dictionaries(dictionary_py)
    prefix = "fr" if args.scenario == "fr_st" else "sp"
    device_name_to_id = dictionaries[f"{prefix}_devices_dict"]
    action_name_to_id = dictionaries[f"{prefix}_actions"]
    device_id_to_name = {int(v): str(k) for k, v in device_name_to_id.items()}
    action_by_device = build_action_by_device(device_name_to_id, action_name_to_id)
    target_stats = dataset_stats(target)
    source_stats = dataset_stats(source)
    old_stats = dataset_stats(old) if old else {"device": Counter(), "action_by_device": defaultdict(Counter), "time_by_device": defaultdict(Counter), "transition": Counter()}

    overused_keys = {
        str(row.get("device_key"))
        for row in guard_report.get("overused_devices", [])
        if row.get("device_key") and float(row.get("ratio", 0.0)) > 1.25
    }
    edge_pairs = extract_structural_edges(hints)
    rng = random.Random(args.seed)
    sequences: list[list[int]] = []
    raw_rows: list[dict[str, Any]] = []
    for index, flat_len in enumerate(length_plan):
        event_count = max(1, flat_len // 4)
        seq = generate_sequence(
            event_count=event_count,
            rng=rng,
            target_stats=target_stats,
            source_stats=source_stats,
            old_stats=old_stats,
            action_by_device=action_by_device,
            device_id_to_name=device_id_to_name,
            overused_keys=overused_keys,
            edge_pairs=edge_pairs,
            scenario=args.scenario,
        )
        sequences.append(seq)
        raw_rows.append({"index": index, "sequence": seq})

    write_jsonl(Path(args.out_jsonl), raw_rows)
    write_pickle(Path(args.out_pkl), sequences)
    metadata = {
        "generator": "codex_gpt55_surrogate",
        "api_llm": False,
        "surrogate_for_smartgen_llm": True,
        "scenario": args.scenario,
        "seed": args.seed,
        "num_sequences": len(sequences),
        "sequence_length": sequence_length,
        "length_plan_unique": sorted(Counter(length_plan).items()),
        "source_prompt": "prompt.txt",
        "package_dir": str(package_dir),
        "guard_mode": "downweight",
        "downweight_factor": 0.25,
        "reweight_mode": "multiplicative",
        "lambda_causal": 1.0,
        "count_reason": count_reason,
        "length_reason": length_reason,
        "target_pkl": str(target_pkl),
        "source_pkl": str(source_pkl),
        "old_generated_pkl": str(Path(args.old_generated_pkl).resolve()) if args.old_generated_pkl else None,
        "schema": schema,
    }
    Path(args.out_metadata).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_metadata).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved generated raw jsonl: {args.out_jsonl}")
    print(f"saved generated pkl: {args.out_pkl}")
    print(f"saved generation metadata: {args.out_metadata}")


def resolve_count(args: argparse.Namespace, old: Sequence[list[int]]) -> tuple[int, str]:
    if old:
        return len(old), "matched_existing_stage3_generated_pkl_count"
    if args.fallback_count:
        return int(args.fallback_count), "user_fallback_count"
    return (200 if args.scenario == "fr_st" else 50), "default_fallback_count"


def resolve_lengths(num_sequences: int, old: Sequence[list[int]], scenario: str) -> tuple[list[int], int, str]:
    if old:
        old_lengths = [len(seq) for seq in old if len(seq) > 0 and len(seq) % 4 == 0]
        if old_lengths:
            plan = [old_lengths[i % len(old_lengths)] for i in range(num_sequences)]
            mode_len = Counter(old_lengths).most_common(1)[0][0]
            return plan, int(mode_len), "cycled_existing_stage3_length_distribution"
    fallback = 40 if scenario == "fr_st" else 16
    return [fallback] * num_sequences, fallback, "default_fallback_length"


def generate_sequence(
    event_count: int,
    rng: random.Random,
    target_stats: dict[str, Any],
    source_stats: dict[str, Any],
    old_stats: dict[str, Any],
    action_by_device: dict[int, list[int]],
    device_id_to_name: dict[int, str],
    overused_keys: set[str],
    edge_pairs: list[tuple[int, int, float]],
    scenario: str,
) -> list[int]:
    devices: list[int] = []
    first_device = weighted_choice_counter(target_stats["device"], rng)
    devices.append(first_device)
    while len(devices) < event_count:
        prev = devices[-1]
        candidates = target_stats["transition_by_source"].get(prev) or source_stats["transition_by_source"].get(prev)
        if candidates and rng.random() < 0.70:
            device = weighted_choice_counter(candidates, rng)
        elif edge_pairs and rng.random() < 0.20:
            src, tgt, _ = weighted_choice(edge_pairs, [max(edge[2], 1e-12) for edge in edge_pairs], rng)
            if src not in devices and len(devices) + 1 < event_count:
                devices.append(src)
            device = tgt
        else:
            device = weighted_choice_counter(target_stats["device"], rng)
        if should_resample_device(device, devices, target_stats, overused_keys, scenario, rng):
            device = weighted_choice_counter(target_stats["device"], rng)
        devices.append(device)

    start_day, start_hour = weighted_choice_counter(target_stats["time"], rng)
    current_slot = int(start_day) * 8 + int(start_hour)
    flat: list[int] = []
    for idx, device in enumerate(devices[:event_count]):
        if idx > 0:
            current_slot += rng.choice([0, 1, 1, 2])
        day = (current_slot // 8) % 7
        hour = current_slot % 8
        action = choose_action(device, target_stats, old_stats, source_stats, action_by_device, rng)
        flat.extend([int(day), int(hour), int(device), int(action)])
    return flat


def should_resample_device(device: int, current_devices: list[int], target_stats: dict[str, Any], overused_keys: set[str], scenario: str, rng: random.Random) -> bool:
    key = f"d:{device}"
    if key not in overused_keys:
        return False
    target_freq = target_stats["device_freq"].get(device, 0.0)
    current_freq = (current_devices.count(device) + 1) / max(len(current_devices) + 1, 1)
    if scenario == "sp_st" and device == 30 and current_freq > max(target_freq * 1.8, 0.08):
        return rng.random() < 0.85
    if current_freq > max(target_freq * 2.0, 0.15):
        return rng.random() < 0.65
    return rng.random() < 0.20


def choose_action(
    device: int,
    target_stats: dict[str, Any],
    old_stats: dict[str, Any],
    source_stats: dict[str, Any],
    action_by_device: dict[int, list[int]],
    rng: random.Random,
) -> int:
    compatible_actions = set(action_by_device.get(device) or [])
    for stats, probability in ((target_stats, 0.70), (old_stats, 0.20), (source_stats, 0.10)):
        counter = stats["action_by_device"].get(device)
        if counter and rng.random() < probability:
            filtered = {action: count for action, count in counter.items() if not compatible_actions or int(action) in compatible_actions}
            if filtered:
                return int(weighted_choice_counter(filtered, rng))
    actions = action_by_device.get(device) or sorted({action for actions in action_by_device.values() for action in actions})
    return int(rng.choice(actions))


def dataset_stats(sequences: Sequence[list[int]]) -> dict[str, Any]:
    device = Counter()
    action_by_device: defaultdict[int, Counter] = defaultdict(Counter)
    time_by_device: defaultdict[int, Counter] = defaultdict(Counter)
    transition = Counter()
    transition_by_source: defaultdict[int, Counter] = defaultdict(Counter)
    time = Counter()
    total_events = 0
    for seq in sequences:
        events = []
        for i in range(0, len(seq), 4):
            if i + 3 >= len(seq):
                continue
            day, hour, dev, action = [int(value) for value in seq[i : i + 4]]
            events.append((day, hour, dev, action))
            device[dev] += 1
            action_by_device[dev][action] += 1
            time_by_device[dev][(day, hour)] += 1
            time[(day, hour)] += 1
            total_events += 1
        for (_, _, src, _), (_, _, tgt, _) in zip(events, events[1:]):
            transition[(src, tgt)] += 1
            transition_by_source[src][tgt] += 1
    device_freq = {dev: count / total_events for dev, count in device.items()} if total_events else {}
    return {
        "device": device,
        "device_freq": device_freq,
        "action_by_device": action_by_device,
        "time_by_device": time_by_device,
        "transition": transition,
        "transition_by_source": transition_by_source,
        "time": time,
    }


def extract_structural_edges(hints: Mapping[str, Any]) -> list[tuple[int, int, float]]:
    rows = []
    for edge in hints.get("edges", []):
        weight = float(edge.get("guarded_causal_strength", edge.get("guarded_weight", edge.get("weight", 0.0))) or 0.0)
        if weight <= 0:
            continue
        src = canonical_device_id(edge.get("source_device_key") or edge.get("source_device") or edge.get("source"))
        tgt = canonical_device_id(edge.get("target_device_key") or edge.get("target_device") or edge.get("target"))
        if src is not None and tgt is not None:
            rows.append((src, tgt, weight))
    return rows


def canonical_device_id(value: Any) -> int | None:
    text = str(value)
    if text.startswith("d:"):
        text = text[2:]
    if text.startswith("device_"):
        text = text[len("device_") :]
    return int(text) if text.isdigit() else None


def build_action_by_device(device_name_to_id: Mapping[str, int], action_name_to_id: Mapping[str, int]) -> dict[int, list[int]]:
    out: defaultdict[int, list[int]] = defaultdict(list)
    for action_name, action_id in action_name_to_id.items():
        device_name = str(action_name).split(":", 1)[0]
        if device_name in device_name_to_id:
            out[int(device_name_to_id[device_name])].append(int(action_id))
    return {device: sorted(set(actions)) for device, actions in out.items()}


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


def weighted_choice_counter(counter: Counter | Mapping[Any, int | float], rng: random.Random) -> Any:
    if not counter:
        raise ValueError("cannot choose from empty counter")
    items = list(counter.items())
    return weighted_choice([item[0] for item in items], [float(item[1]) for item in items], rng)


def weighted_choice(items: Sequence[Any], weights: Sequence[float], rng: random.Random) -> Any:
    total = sum(max(float(weight), 0.0) for weight in weights)
    if total <= 0:
        return rng.choice(list(items))
    threshold = rng.random() * total
    cursor = 0.0
    for item, weight in zip(items, weights):
        cursor += max(float(weight), 0.0)
        if cursor >= threshold:
            return item
    return items[-1]


def load_pickle(path: Path) -> list[list[int]]:
    install_numpy_pickle_compat()
    with open(path, "rb") as f:
        raw = pickle.load(f)
    out = []
    for item in raw:
        if hasattr(item, "tolist"):
            item = item.tolist()
        if isinstance(item, tuple):
            item = next((part for part in item if isinstance(part, Sequence) and not isinstance(part, (str, bytes))), item)
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            try:
                out.append([int(value) for value in item])
            except Exception:
                continue
    return out


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_pickle(path: Path, sequences: Sequence[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(list(sequences), f)


if __name__ == "__main__":
    main()
