#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.experiment_matrix import SEEDS, experiment_grid, matrix_item, matrix_stage_dir
from scripts.validate_and_pack_gpt55_generation import load_vocab

OUT_ROOT = REPO_ROOT / "outputs" / "main_experiment"
PACKAGE_INDEX = OUT_ROOT / "gpt55_generation_packages" / "generation_package_index.json"
DICTIONARY = REPO_ROOT / "causal_smart_home" / "resources" / "gen_data" / "dictionary.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the full CausalSmartHome matrix JSONL with Codex-authored package synthesis.")
    parser.add_argument("--matrix", default="all", choices=["all"])
    parser.add_argument("--input-index", type=Path, default=PACKAGE_INDEX)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT / "gpt55_generation")
    parser.add_argument("--run-id", default=f"codex_full_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--expected-count", type=int, default=50)
    parser.add_argument("--events-per-sequence", type=int, default=4)
    parser.add_argument("--generate-missing-only", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--backup-existing", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def matrix_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {(row["dataset"], row["scenario"], int(row["seed"])): row for row in rows}


def output_dir(output_root: Path, dataset: str, scenario: str, seed: int) -> Path:
    root_name = output_root.name
    root_parent = output_root.parent
    if root_name == "gpt55_generation":
        return matrix_stage_dir(root_parent, root_name, dataset, scenario, seed)
    return output_root / dataset / scenario / f"seed{seed}"


def action_by_device(dataset: str) -> dict[int, list[int]]:
    vocab = load_vocab(DICTIONARY, dataset)
    return {int(device): [int(action) for action in actions] for device, actions in vocab["action_by_device"].items()}


def parse_device_key(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    text = str(value)
    if text.startswith("d:"):
        text = text[2:]
    try:
        return int(text)
    except Exception:
        return None


def read_overused_devices(guard_report: dict[str, Any]) -> set[int]:
    out: set[int] = set()
    for item in guard_report.get("overused_devices", []):
        device = parse_device_key(item.get("device_key"))
        if device is not None and item.get("overused"):
            out.add(device)
    return out


def target_device_weights(guard_report: dict[str, Any], legal_devices: set[int]) -> dict[int, float]:
    weights: dict[int, float] = {}
    for item in guard_report.get("overused_devices", []):
        device = parse_device_key(item.get("device_key"))
        if device in legal_devices:
            weights[int(device)] = max(float(item.get("target_freq") or 0.0), 0.001)
    for edge in guard_report.get("edges", []):
        for key in ("source_device_key", "target_device_key", "source", "target"):
            device = parse_device_key(edge.get(key))
            if device in legal_devices:
                weights.setdefault(int(device), 0.01)
    for device in legal_devices:
        weights.setdefault(device, 0.005)
    return weights


def choose_weighted(rng: random.Random, weighted: list[tuple[Any, float]]) -> Any:
    total = sum(max(float(weight), 0.0) for _, weight in weighted)
    if total <= 0:
        return weighted[rng.randrange(len(weighted))][0]
    pick = rng.random() * total
    running = 0.0
    for item, weight in weighted:
        running += max(float(weight), 0.0)
        if running >= pick:
            return item
    return weighted[-1][0]


def edge_candidates(hints: dict[str, Any], guard_report: dict[str, Any], legal_devices: set[int]) -> list[dict[str, Any]]:
    overused = read_overused_devices(guard_report)
    candidates = []
    for edge in hints.get("edges", []):
        source = parse_device_key(edge.get("source_device", edge.get("source_device_key")))
        target = parse_device_key(edge.get("target_device", edge.get("target_device_key")))
        if source not in legal_devices or target not in legal_devices:
            continue
        guard_action = str(edge.get("guard_action", "keep"))
        if guard_action == "suppress" and target in overused:
            continue
        score = float(edge.get("final_score") or edge.get("transition_score") or edge.get("guarded_causal_strength") or 0.01)
        if source == target:
            score *= 0.8
        candidates.append({"source": int(source), "target": int(target), "score": max(score, 0.01), "guard_action": guard_action})
    if candidates:
        return candidates
    for edge in guard_report.get("edges", []):
        source = parse_device_key(edge.get("source_device_key", edge.get("source")))
        target = parse_device_key(edge.get("target_device_key", edge.get("target")))
        if source in legal_devices and target in legal_devices and edge.get("guard_action") != "suppress":
            candidates.append({"source": int(source), "target": int(target), "score": max(float(edge.get("guarded_weight") or 0.01), 0.01), "guard_action": edge.get("guard_action", "keep")})
    return candidates


def generate_sequences(
    dataset: str,
    scenario: str,
    seed: int,
    run_id: str,
    package_dir: Path,
    count: int,
    events_per_sequence: int,
) -> list[dict[str, Any]]:
    actions = action_by_device(dataset)
    legal_devices = {device for device, device_actions in actions.items() if device_actions}
    guard_report = load_json(package_dir / "guard_report.json")
    hints = load_json(package_dir / "guarded_reweighted_gss_hints.json")
    edges = edge_candidates(hints, guard_report, legal_devices)
    if not edges:
        raise ValueError(f"no usable guarded edges in package: {package_dir}")
    by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        by_source[edge["source"]].append(edge)
    device_weights = target_device_weights(guard_report, legal_devices)
    edge_weighted = [(edge, edge["score"]) for edge in edges]
    device_weighted = list(device_weights.items())
    rng = random.Random(stable_seed(run_id, dataset, scenario, seed))
    rows = []
    seen: set[tuple[int, ...]] = set()
    attempts = 0
    while len(rows) < count and attempts < count * 50:
        attempts += 1
        first_edge = choose_weighted(rng, edge_weighted)
        devices = [first_edge["source"], first_edge["target"]]
        while len(devices) < events_per_sequence:
            previous = devices[-1]
            options = by_source.get(previous)
            if options and rng.random() < 0.82:
                edge = choose_weighted(rng, [(item, item["score"]) for item in options])
                devices.append(edge["target"])
            else:
                devices.append(choose_weighted(rng, device_weighted))
        day = rng.randrange(7)
        hour = rng.randrange(8)
        flat: list[int] = []
        for index, device in enumerate(devices[:events_per_sequence]):
            legal_actions = actions.get(int(device)) or actions[choose_weighted(rng, device_weighted)]
            action = int(legal_actions[rng.randrange(len(legal_actions))])
            flat.extend([(day + (hour + index) // 8) % 7, (hour + index) % 8, int(device), action])
        key = tuple(flat)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "sequence_id": f"codex_{dataset}_{scenario}_seed{seed}_{len(rows):03d}",
                "sequence": flat,
                "notes": "Codex package synthesis using guarded causal-reweighted GSS hints, target guard constraints, and seeded variation.",
            }
        )
    if len(rows) != count:
        raise ValueError(f"generated {len(rows)} unique rows, expected {count}")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def maybe_backup(out_dir: Path, run_id: str) -> str | None:
    if not out_dir.exists() or not any(out_dir.iterdir()):
        return None
    backup_dir = out_dir.parent / f"{out_dir.name}.backup_before_{run_id}"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(out_dir, backup_dir)
    return str(backup_dir.resolve())


def legacy_output_dir(output_root: Path, dataset: str, scenario: str, seed: int) -> Path:
    return output_root / f"{dataset}_{scenario}" / f"seed{seed}"


def existing_complete(out_dir: Path) -> bool:
    return (out_dir / "generated_gpt55_clean.jsonl").exists() and (out_dir / "generated_gpt55_clean.pkl").exists()


def cell_log(status: str, message: str, payload: dict[str, Any]) -> str:
    lines = [
        f"# Codex Generation Log: {payload['dataset'].upper()}-{payload['scenario'].upper()} seed{payload['seed']}",
        "",
        f"- status: {status}",
        f"- message: {message}",
        f"- generation_model_actual: {payload.get('generation_model_actual')}",
        f"- legacy_generation_name: {payload.get('legacy_generation_name')}",
        f"- run_id: {payload.get('run_id')}",
        f"- input_package_path: {payload.get('input_package_path')}",
        f"- output_jsonl_path: {payload.get('output_jsonl_path')}",
    ]
    if payload.get("backup_dir"):
        lines.append(f"- backup_dir: {payload['backup_dir']}")
    if payload.get("legacy_backup_dir"):
        lines.append(f"- legacy_backup_dir: {payload['legacy_backup_dir']}")
    if payload.get("error"):
        lines.append(f"- error: {payload['error']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    index_rows = load_json(args.input_index.resolve())
    indexed = matrix_index(index_rows)
    manifest_path = OUT_ROOT / "generation_run_manifest.json"
    report_path = OUT_ROOT / "generation_run_report.md"
    started_at = datetime.now().isoformat(timespec="seconds")
    manifest = {
        "run_id": args.run_id,
        "legacy_generation_name": "gpt55_generation",
        "generation_model_actual": "Codex",
        "matrix_cells": 27,
        "seeds": SEEDS,
        "status": "STARTED",
        "started_at": started_at,
        "input_index": str(args.input_index.resolve()),
        "output_root": str(args.output_root.resolve()),
        "cells": [],
    }
    write_json(manifest_path, manifest)

    for item in experiment_grid():
        for seed in SEEDS:
            row = indexed[(item.dataset, item.scenario, seed)]
            package_dir = Path(row["package_dir"]).resolve()
            out_dir = output_dir(args.output_root.resolve(), item.dataset, item.scenario, seed)
            jsonl_path = out_dir / "generated_gpt55_clean.jsonl"
            metadata_path = out_dir / "generation_metadata.json"
            log_path = out_dir / "generation_log.md"
            backup_dir = None
            metadata = {
                "run_id": args.run_id,
                "dataset": item.dataset,
                "scenario": item.scenario,
                "seed": seed,
                "source_context": matrix_item(item.dataset, item.scenario).source_context,
                "target_context": matrix_item(item.dataset, item.scenario).target_context,
                "generation_model_actual": "Codex",
                "legacy_generation_name": "gpt55_generation",
                "input_package_path": str(package_dir),
                "output_jsonl_path": str(jsonl_path.resolve()),
                "schema_version": "flat_quadruples_v1",
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            try:
                if args.generate_missing_only and existing_complete(out_dir):
                    metadata["status"] = "SKIPPED_EXISTING_COMPLETE"
                    write_json(metadata_path, metadata)
                    log_path.write_text(cell_log("SKIPPED_EXISTING_COMPLETE", "Existing validated generation was preserved.", metadata), encoding="utf-8")
                    manifest["cells"].append({**metadata, "status": "SKIPPED_EXISTING_COMPLETE"})
                    continue
                if jsonl_path.exists() and not args.overwrite_existing and not args.generate_missing_only:
                    metadata["status"] = "FAILED"
                    metadata["error"] = "existing JSONL present; pass --overwrite-existing to replace"
                    write_json(metadata_path, metadata)
                    log_path.write_text(cell_log("FAILED", metadata["error"], metadata), encoding="utf-8")
                    manifest["cells"].append(metadata)
                    continue
                if args.backup_existing or (args.overwrite_existing and out_dir.exists()):
                    backup_dir = maybe_backup(out_dir, args.run_id)
                    metadata["backup_dir"] = backup_dir
                    legacy_dir = legacy_output_dir(args.output_root.resolve(), item.dataset, item.scenario, seed)
                    if legacy_dir != out_dir and legacy_dir.exists():
                        metadata["legacy_backup_dir"] = maybe_backup(legacy_dir, args.run_id)
                sequences = generate_sequences(item.dataset, item.scenario, seed, args.run_id, package_dir, args.expected_count, args.events_per_sequence)
                write_jsonl(jsonl_path, sequences)
                write_jsonl(out_dir / "generated.jsonl", sequences)
                metadata.update(
                    {
                        "status": "GENERATED",
                        "expected_count": args.expected_count,
                        "events_per_sequence": args.events_per_sequence,
                        "num_generated": len(sequences),
                    }
                )
                write_json(metadata_path, metadata)
                log_path.write_text(cell_log("GENERATED", "Generated Codex-authored JSONL from the fixed package artifacts.", metadata), encoding="utf-8")
                manifest["cells"].append(metadata)
            except Exception as exc:
                metadata["status"] = "FAILED"
                metadata["error"] = str(exc)
                write_json(metadata_path, metadata)
                log_path.write_text(cell_log("FAILED", str(exc), metadata), encoding="utf-8")
                manifest["cells"].append(metadata)

    failed = [cell for cell in manifest["cells"] if cell["status"] == "FAILED"]
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["status"] = "FAILED" if failed else "COMPLETE"
    write_json(manifest_path, manifest)
    lines = [
        "# Codex Generation Run Report",
        "",
        f"- run_id: {args.run_id}",
        "- generation_model_actual: Codex",
        "- legacy_generation_name: gpt55_generation",
        f"- status: {manifest['status']}",
        f"- generated: {sum(1 for cell in manifest['cells'] if cell['status'] == 'GENERATED')}",
        f"- skipped_existing_complete: {sum(1 for cell in manifest['cells'] if cell['status'] == 'SKIPPED_EXISTING_COMPLETE')}",
        f"- failed: {len(failed)}",
        "",
        "| dataset | scenario | seed | status | jsonl |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cell in manifest["cells"]:
        lines.append(f"| {cell['dataset']} | {cell['scenario']} | {cell['seed']} | {cell['status']} | {cell.get('output_jsonl_path', '')} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failed:
        raise SystemExit(f"Codex generation failed for {len(failed)} cell(s); see {manifest_path}")
    print(f"generation run status: {manifest['status']}")
    print(f"manifest: {manifest_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
