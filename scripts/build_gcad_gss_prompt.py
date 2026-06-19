#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.causal_gss import (
    build_causal_hints_payload,
    format_causal_hints_for_prompt,
    learn_device_gcad_prior,
    load_id_name_mapping,
    load_pickle_sequences,
    map_device_edges,
    render_edges_markdown,
)
from causal_smart_home.causal_prompt_adapter import (
    enhance_prompt_with_causal_hints,
    load_prompt,
    render_prompt_diff,
)


def jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def resolve_out_prompt(value: str | Path) -> Path:
    path = Path(value)
    if path.suffix:
        return path
    return path / "enhanced_prompt.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an offline FR-ST GCAD-GSS enhanced SmartGen prompt without calling an LLM."
    )
    parser.add_argument("--source-train-pkl", required=True, help="Source real normal SmartGen/SmartGuard numeric pkl.")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--smartgen-original-prompt", help="Original SmartGen prompt text file.")
    prompt_group.add_argument("--smartgen-prompt-template", help="SmartGen prompt template text file.")
    parser.add_argument("--device-dict", help="Optional JSON or dictionary.py device mapping.")
    parser.add_argument("--action-dict", help="Optional JSON or dictionary.py action mapping; recorded for traceability.")
    parser.add_argument("--out-prompt", required=True, help="Output prompt file or output directory.")
    parser.add_argument("--level", choices=["device"], default="device")
    parser.add_argument("--top-k-edges", type=int, default=20)
    parser.add_argument("--lag", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--sparse-threshold", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--scenario", default="fr_st", help="Trace label; this wrapper currently targets FR-ST only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scenario != "fr_st":
        raise ValueError("this wrapper is currently scoped to FR-ST; pass --scenario fr_st")

    source_train_pkl = Path(args.source_train_pkl).resolve()
    prompt_path = Path(args.smartgen_original_prompt or args.smartgen_prompt_template).resolve()
    out_prompt = resolve_out_prompt(args.out_prompt).resolve()
    out_dir = out_prompt.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = load_pickle_sequences(source_train_pkl)
    prior = learn_device_gcad_prior(
        sequences,
        lag=args.lag,
        epochs=args.epochs,
        sparse_threshold=args.sparse_threshold,
        batch_size=args.batch_size,
        hidden=args.hidden,
        sample_limit=args.sample_limit,
    )

    device_mapping = load_id_name_mapping(args.device_dict, preferred_names=("fr_devices_dict", "device_dict"))
    action_mapping = load_id_name_mapping(args.action_dict, preferred_names=("fr_actions", "action_dict")) if args.action_dict else {}
    edges = map_device_edges(prior, device_mapping, top_k_edges=args.top_k_edges)
    payload = build_causal_hints_payload(edges, prior, source_train_pkl=source_train_pkl, level=args.level)
    payload["scenario"] = args.scenario
    payload["prompt_source"] = str(prompt_path)
    payload["dictionary"] = {
        "device_dict": str(Path(args.device_dict).resolve()) if args.device_dict else None,
        "action_dict": str(Path(args.action_dict).resolve()) if args.action_dict else None,
        "num_device_names": len(device_mapping),
        "num_action_names": len(action_mapping),
    }

    original_prompt = load_prompt(prompt_path)
    hints_text = format_causal_hints_for_prompt(payload)
    result = enhance_prompt_with_causal_hints(original_prompt, hints_text)

    out_prompt.write_text(result.enhanced_prompt, encoding="utf-8")
    (out_dir / "causal_hints.json").write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "top_causal_edges.md").write_text(render_edges_markdown(edges, payload), encoding="utf-8")
    (out_dir / "prompt_diff.md").write_text(render_prompt_diff(result.original_prompt, result.enhanced_prompt), encoding="utf-8")
    (out_dir / "run_config.json").write_text(json.dumps(jsonable(vars(args)), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"loaded source normal sequences: {len(sequences)}")
    print(f"learned device channels: {len(prior.channel_to_key)}")
    print(f"selected causal edges: {len(edges)}")
    print(f"inserted causal hints after: {result.insertion_label}")
    print(f"saved {out_prompt}")
    print(f"saved {out_dir / 'causal_hints.json'}")
    print(f"saved {out_dir / 'top_causal_edges.md'}")
    print(f"saved {out_dir / 'prompt_diff.md'}")


if __name__ == "__main__":
    main()
