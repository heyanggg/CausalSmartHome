#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.schema import BehaviorSequence


CSH_ROOT = REPO_ROOT
OUT_ROOT = CSH_ROOT / "outputs/gcad_gss"
SMARTGEN_ROOT = Path("/home/heyang/projects/SmartGen/SmartGen")
DEFAULT_TARGET_REAL = Path("/home/heyang/projects/SmartGen/parameter_study/test/sp/spring/split_test.pkl")
DEFAULT_RUNS = [
    "seed2024=sp_st_codex_calibrated_seed2024",
    "seed2025=sp_st_codex_calibrated_seed2025",
    "seed2026=sp_st_codex_calibrated_seed2026",
]
EPS = 1e-12


def load_pickle(path: str | Path) -> Any:
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
    with open(path, "rb") as f:
        return pickle.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(value) for value in obj]
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def load_sequences(path: str | Path) -> list[BehaviorSequence]:
    raw = load_pickle(path)
    sequences: list[BehaviorSequence] = []
    if not isinstance(raw, Iterable):
        raise ValueError(f"expected iterable pickle: {path}")
    for index, item in enumerate(raw):
        if isinstance(item, np.ndarray):
            item = item.tolist()
        if isinstance(item, tuple):
            for part in item:
                if isinstance(part, np.ndarray):
                    part = part.tolist()
                if isinstance(part, list) and all(isinstance(value, (int, np.integer)) for value in part):
                    item = part
                    break
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)):
            continue
        if len(item) == 0 or len(item) % 4 != 0:
            continue
        try:
            flat = [int(value) for value in item]
        except Exception:
            continue
        sequences.append(BehaviorSequence.from_flat_numeric(flat, sequence_id=str(index)))
    return sequences


def load_dictionary(path: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("stage3_sp_dictionary", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load SmartGen dictionary from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        "device_id_to_name": {int(value): key for key, value in module.sp_devices_dict.items()},
        "action_id_to_name": {int(value): key for key, value in module.sp_actions.items()},
    }


def counts(sequences: Sequence[BehaviorSequence], level: str) -> Counter[str]:
    out: Counter[str] = Counter()
    for seq in sequences:
        for event in seq:
            out[event.key(level)] += 1
    return out


def frequencies(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in counter.items()}


def kl_divergence_array(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def js_divergence(left: Counter[str], right: Counter[str]) -> float | None:
    keys = sorted(set(left) | set(right))
    left_total = sum(left.values())
    right_total = sum(right.values())
    if not keys or left_total == 0 or right_total == 0:
        return None
    p = np.asarray([left.get(key, 0) / left_total for key in keys], dtype=np.float64)
    q = np.asarray([right.get(key, 0) / right_total for key in keys], dtype=np.float64)
    m = 0.5 * (p + q)
    return float(0.5 * kl_divergence_array(p, m) + 0.5 * kl_divergence_array(q, m))


def key_name(key: str, mappings: dict[str, Any], level: str) -> str:
    prefix = "d:" if level == "device" else "a:"
    if not key.startswith(prefix):
        return key
    try:
        ident = int(key[len(prefix) :])
    except ValueError:
        return key
    if level == "device":
        return mappings["device_id_to_name"].get(ident, key)
    return mappings["action_id_to_name"].get(ident, key)


def order_stats(sequences: Sequence[BehaviorSequence], source_key: str, target_key: str) -> dict[str, Any]:
    total = len(sequences)
    both = 0
    source_only = 0
    target_only = 0
    source_before_target = 0
    target_before_source = 0
    same_first_position = 0
    for seq in sequences:
        source_positions = [idx for idx, event in enumerate(seq) if event.key("device") == source_key]
        target_positions = [idx for idx, event in enumerate(seq) if event.key("device") == target_key]
        if source_positions and target_positions:
            both += 1
            first_source = min(source_positions)
            first_target = min(target_positions)
            if first_source < first_target:
                source_before_target += 1
            elif first_target < first_source:
                target_before_source += 1
            else:
                same_first_position += 1
        elif source_positions:
            source_only += 1
        elif target_positions:
            target_only += 1
    return {
        "sequence_count": total,
        "both": both,
        "source_only": source_only,
        "target_only": target_only,
        "neither": total - both - source_only - target_only,
        "cooccurrence_rate": both / total if total else 0.0,
        "source_before_target": source_before_target,
        "target_before_source": target_before_source,
        "same_first_position": same_first_position,
        "source_before_rate_among_both": source_before_target / both if both else None,
    }


def distribution_delta_rows(
    original: Sequence[BehaviorSequence],
    enhanced: Sequence[BehaviorSequence],
    target: Sequence[BehaviorSequence],
    mappings: dict[str, Any],
    level: str,
    limit: int,
) -> dict[str, Any]:
    original_counts = counts(original, level)
    enhanced_counts = counts(enhanced, level)
    target_counts = counts(target, level)
    original_freq = frequencies(original_counts)
    enhanced_freq = frequencies(enhanced_counts)
    target_freq = frequencies(target_counts)
    rows: list[dict[str, Any]] = []
    for key in sorted(set(original_freq) | set(enhanced_freq) | set(target_freq)):
        o = original_freq.get(key, 0.0)
        e = enhanced_freq.get(key, 0.0)
        t = target_freq.get(key, 0.0)
        rows.append(
            {
                "key": key,
                "name": key_name(key, mappings, level),
                "original_freq": o,
                "enhanced_freq": e,
                "target_freq": t,
                "enhanced_minus_original": e - o,
                "original_abs_gap_to_target": abs(o - t),
                "enhanced_abs_gap_to_target": abs(e - t),
                "gap_change": abs(e - t) - abs(o - t),
                "original_count": int(original_counts.get(key, 0)),
                "enhanced_count": int(enhanced_counts.get(key, 0)),
                "target_count": int(target_counts.get(key, 0)),
            }
        )
    worsened = sorted(rows, key=lambda row: row["gap_change"], reverse=True)[:limit]
    improved = sorted(rows, key=lambda row: row["gap_change"])[:limit]
    increased = sorted(rows, key=lambda row: row["enhanced_minus_original"], reverse=True)[:limit]
    decreased = sorted(rows, key=lambda row: row["enhanced_minus_original"])[:limit]
    return {
        "level": level,
        "original_js_to_target": js_divergence(original_counts, target_counts),
        "enhanced_js_to_target": js_divergence(enhanced_counts, target_counts),
        "delta_js_to_target": (
            js_divergence(enhanced_counts, target_counts) - js_divergence(original_counts, target_counts)
            if js_divergence(original_counts, target_counts) is not None and js_divergence(enhanced_counts, target_counts) is not None
            else None
        ),
        "top_worsened_gap_to_target": worsened,
        "top_improved_gap_to_target": improved,
        "top_increased_by_enhanced": increased,
        "top_decreased_by_enhanced": decreased,
    }


def edge_endpoint_row(
    edge: dict[str, Any],
    original: Sequence[BehaviorSequence],
    enhanced: Sequence[BehaviorSequence],
    target: Sequence[BehaviorSequence],
    mappings: dict[str, Any],
) -> dict[str, Any]:
    original_freq = frequencies(counts(original, "device"))
    enhanced_freq = frequencies(counts(enhanced, "device"))
    target_freq = frequencies(counts(target, "device"))
    source = str(edge["source"])
    target_key = str(edge["target"])
    source_original = original_freq.get(source, 0.0)
    source_enhanced = enhanced_freq.get(source, 0.0)
    source_target = target_freq.get(source, 0.0)
    target_original = original_freq.get(target_key, 0.0)
    target_enhanced = enhanced_freq.get(target_key, 0.0)
    target_target = target_freq.get(target_key, 0.0)
    original_order = order_stats(original, source, target_key)
    enhanced_order = order_stats(enhanced, source, target_key)
    return {
        "source": source,
        "target": target_key,
        "source_name": key_name(source, mappings, "device"),
        "target_name": key_name(target_key, mappings, "device"),
        "weight": float(edge.get("weight", 0.0)),
        "source_enhanced_minus_original_freq": source_enhanced - source_original,
        "target_enhanced_minus_original_freq": target_enhanced - target_original,
        "source_original_freq": source_original,
        "source_enhanced_freq": source_enhanced,
        "source_target_freq": source_target,
        "target_original_freq": target_original,
        "target_enhanced_freq": target_enhanced,
        "target_target_freq": target_target,
        "source_gap_change": abs(source_enhanced - source_target) - abs(source_original - source_target),
        "target_gap_change": abs(target_enhanced - target_target) - abs(target_original - target_target),
        "original_order": original_order,
        "enhanced_order": enhanced_order,
        "delta_cooccurrence_rate": enhanced_order["cooccurrence_rate"] - original_order["cooccurrence_rate"],
        "delta_source_before_rate_among_both": (
            enhanced_order["source_before_rate_among_both"] - original_order["source_before_rate_among_both"]
            if enhanced_order["source_before_rate_among_both"] is not None
            and original_order["source_before_rate_among_both"] is not None
            else None
        ),
    }


def avg(values: list[float]) -> float | None:
    return mean(values) if values else None


def sd(values: list[float]) -> float | None:
    return stdev(values) if len(values) > 1 else 0.0 if len(values) == 1 else None


def aggregate_distribution(runs: list[dict[str, Any]], level: str, bucket: str, limit: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        for row in run["distribution"][level][bucket]:
            grouped[row["key"]].append(row)
    out = []
    for key, rows in grouped.items():
        out.append(
            {
                "key": key,
                "name": rows[0]["name"],
                "mean_gap_change": avg([row["gap_change"] for row in rows]),
                "std_gap_change": sd([row["gap_change"] for row in rows]),
                "mean_enhanced_minus_original": avg([row["enhanced_minus_original"] for row in rows]),
                "mean_original_freq": avg([row["original_freq"] for row in rows]),
                "mean_enhanced_freq": avg([row["enhanced_freq"] for row in rows]),
                "mean_target_freq": avg([row["target_freq"] for row in rows]),
                "seed_count": len(rows),
            }
        )
    reverse = bucket in {"top_worsened_gap_to_target", "top_increased_by_enhanced"}
    sort_key = "mean_gap_change" if "gap" in bucket else "mean_enhanced_minus_original"
    return sorted(out, key=lambda row: row[sort_key] if row[sort_key] is not None else 0.0, reverse=reverse)[:limit]


def aggregate_edges(runs: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        for row in run["edge_endpoint_rows"]:
            grouped[(row["source"], row["target"])].append(row)
    out = []
    for (_source, _target), rows in grouped.items():
        out.append(
            {
                "source": rows[0]["source"],
                "target": rows[0]["target"],
                "source_name": rows[0]["source_name"],
                "target_name": rows[0]["target_name"],
                "mean_weight": avg([row["weight"] for row in rows]),
                "mean_source_gap_change": avg([row["source_gap_change"] for row in rows]),
                "mean_target_gap_change": avg([row["target_gap_change"] for row in rows]),
                "mean_delta_cooccurrence_rate": avg([row["delta_cooccurrence_rate"] for row in rows]),
                "mean_delta_source_before_rate_among_both": avg(
                    [
                        row["delta_source_before_rate_among_both"]
                        for row in rows
                        if row["delta_source_before_rate_among_both"] is not None
                    ]
                ),
                "mean_source_enhanced_minus_original_freq": avg(
                    [row["source_enhanced_minus_original_freq"] for row in rows]
                ),
                "mean_target_enhanced_minus_original_freq": avg(
                    [row["target_enhanced_minus_original_freq"] for row in rows]
                ),
                "seed_count": len(rows),
            }
        )
    return sorted(
        out,
        key=lambda row: (
            (row["mean_source_gap_change"] or 0.0)
            + (row["mean_target_gap_change"] or 0.0)
            + abs(row["mean_delta_cooccurrence_rate"] or 0.0)
        ),
        reverse=True,
    )[:limit]


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(payload: dict[str, Any]) -> str:
    aggregate = payload["aggregate"]
    lines = [
        "# SP-ST Causal Edge Diagnostic",
        "",
        "This diagnostic compares original vs GCAD-GSS enhanced SP-ST TOF outputs across seeds 2024, 2025, and 2026.",
        "",
        "## Summary",
        "",
        f"- Device JS delta mean: `{fmt(aggregate['js_delta_mean']['device'])}`.",
        f"- Action JS delta mean: `{fmt(aggregate['js_delta_mean']['action'])}`.",
        "- Positive JS delta means enhanced moved farther from the SP spring target distribution.",
        "- The SP GCAD prior is very low-weight; the useful signal is visible in causal coverage/low evidence, but endpoint distribution pressure is not aligned with the target distribution.",
        "",
        "## Edges Most Associated With Distribution/Order Shifts",
        "",
        "| source | target | weight | source gap Δ | target gap Δ | cooccur Δ | order-rate Δ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate["edge_endpoint_rows_top"]:
        lines.append(
            f"| {row['source_name']} | {row['target_name']} | {fmt(row['mean_weight'])} | "
            f"{fmt(row['mean_source_gap_change'])} | {fmt(row['mean_target_gap_change'])} | "
            f"{fmt(row['mean_delta_cooccurrence_rate'])} | {fmt(row['mean_delta_source_before_rate_among_both'])} |"
        )

    for level, title in (("device", "Device"), ("action", "Action")):
        lines.extend(["", f"## Top {title} Gaps Worsened By Enhanced", ""])
        lines.append("| name | gap Δ | enhanced-original freq | original freq | enhanced freq | target freq |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in aggregate["distribution"][level]["top_worsened_gap_to_target"]:
            lines.append(
                f"| {row['name']} | {fmt(row['mean_gap_change'])} | "
                f"{fmt(row['mean_enhanced_minus_original'])} | {fmt(row['mean_original_freq'])} | "
                f"{fmt(row['mean_enhanced_freq'])} | {fmt(row['mean_target_freq'])} |"
            )
        lines.extend(["", f"## Top {title} Gaps Improved By Enhanced", ""])
        lines.append("| name | gap Δ | enhanced-original freq | original freq | enhanced freq | target freq |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in aggregate["distribution"][level]["top_improved_gap_to_target"]:
            lines.append(
                f"| {row['name']} | {fmt(row['mean_gap_change'])} | "
                f"{fmt(row['mean_enhanced_minus_original'])} | {fmt(row['mean_original_freq'])} | "
                f"{fmt(row['mean_enhanced_freq'])} | {fmt(row['mean_target_freq'])} |"
            )

    lines.extend(["", "## Diagnosis", ""])
    lines.extend(payload["diagnosis"])
    lines.extend(["", "## Artifacts", ""])
    for key, value in payload["artifacts"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Diagnose SP-ST GCAD-GSS causal edge effects")
    parser.add_argument("--run", action="append", default=[], metavar="NAME=TAG")
    parser.add_argument("--target-real-pkl", type=Path, default=DEFAULT_TARGET_REAL)
    parser.add_argument("--smartgen-dictionary", type=Path, default=SMARTGEN_ROOT / "dictionary.py")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--out-json", type=Path, default=OUT_ROOT / "sp_st_causal_edge_diagnostic.json")
    parser.add_argument("--out-md", type=Path, default=OUT_ROOT / "sp_st_causal_edge_diagnostic.md")
    parser.add_argument("--out-edge-csv", type=Path, default=OUT_ROOT / "sp_st_causal_edge_diagnostic_edges.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mappings = load_dictionary(args.smartgen_dictionary)
    target_sequences = load_sequences(args.target_real_pkl)
    runs_arg = args.run or DEFAULT_RUNS
    run_payloads: list[dict[str, Any]] = []

    for item in runs_arg:
        name, tag = item.split("=", 1)
        root = OUT_ROOT / tag
        original_tof = root / "sp_st_original/smartgen_tof.pkl"
        enhanced_tof = root / "sp_st_enhanced/smartgen_tof.pkl"
        hints_path = root / "sp_st_prompt_check/causal_hints.json"
        if not original_tof.exists() or not enhanced_tof.exists() or not hints_path.exists():
            raise FileNotFoundError(f"missing SP-ST run artifact for {name}={tag}")
        original_sequences = load_sequences(original_tof)
        enhanced_sequences = load_sequences(enhanced_tof)
        hints = json.loads(hints_path.read_text(encoding="utf-8"))
        edges = list(hints.get("top_causal_edges") or [])[: args.top_k]
        distribution = {
            "device": distribution_delta_rows(
                original_sequences, enhanced_sequences, target_sequences, mappings, "device", args.top_k
            ),
            "action": distribution_delta_rows(
                original_sequences, enhanced_sequences, target_sequences, mappings, "action", args.top_k
            ),
        }
        edge_rows = [
            edge_endpoint_row(edge, original_sequences, enhanced_sequences, target_sequences, mappings)
            for edge in edges
        ]
        run_payloads.append(
            {
                "repeat": name,
                "tag": tag,
                "paths": {
                    "original_tof": str(original_tof),
                    "enhanced_tof": str(enhanced_tof),
                    "causal_hints": str(hints_path),
                },
                "sequence_counts": {
                    "original": len(original_sequences),
                    "enhanced": len(enhanced_sequences),
                    "target": len(target_sequences),
                },
                "distribution": distribution,
                "edge_endpoint_rows": edge_rows,
            }
        )

    aggregate_distribution_payload = {
        level: {
            bucket: aggregate_distribution(run_payloads, level, bucket, args.top_k)
            for bucket in (
                "top_worsened_gap_to_target",
                "top_improved_gap_to_target",
                "top_increased_by_enhanced",
                "top_decreased_by_enhanced",
            )
        }
        for level in ("device", "action")
    }
    js_delta_mean = {
        level: avg([run["distribution"][level]["delta_js_to_target"] for run in run_payloads])
        for level in ("device", "action")
    }
    edge_top = aggregate_edges(run_payloads, args.top_k)
    diagnosis = [
        "- Enhanced consistently reduces low-evidence in Stage 3A, but this diagnostic shows the injected edge endpoints can increase gap-to-target for several high-impact devices/actions.",
        "- The most suspicious pattern is not one bad seed; the mean action/device JS deltas are both positive across the three-seed aggregate.",
        "- The next ablation should reduce endpoint pressure before adding new environments: try smaller top-k values and endpoint filters for edges whose source or target is already overrepresented relative to SP spring target data.",
    ]
    payload = {
        "settings": {
            "runs": runs_arg,
            "target_real_pkl": str(args.target_real_pkl),
            "smartgen_dictionary": str(args.smartgen_dictionary),
            "top_k": args.top_k,
        },
        "runs": run_payloads,
        "aggregate": {
            "js_delta_mean": js_delta_mean,
            "distribution": aggregate_distribution_payload,
            "edge_endpoint_rows_top": edge_top,
        },
        "diagnosis": diagnosis,
        "artifacts": {
            "json": str(args.out_json),
            "markdown": str(args.out_md),
            "edge_csv": str(args.out_edge_csv),
        },
    }
    write_json(args.out_json, payload)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(build_markdown(payload), encoding="utf-8")
    write_csv(args.out_edge_csv, edge_top)
    print(f"diagnostic_json: {args.out_json}")
    print(f"diagnostic_md: {args.out_md}")
    print(f"diagnostic_edge_csv: {args.out_edge_csv}")


if __name__ == "__main__":
    main()
