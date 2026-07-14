#!/usr/bin/env python
"""构建生成阶段使用的 GCAD + Gen GSS prompt 产物。

输入：
    源上下文 normal Gen pkl、target normal pkl，以及可选 causal prior。

输出：resolved prior JSON、causal-reweighted GSS hints 和供 Codex 生成使用的 prompt。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.causal_relation_prior_source import ResolvedCausalRelationPrior, resolve_causal_relation_prior
from causal_smart_home.causal.adaptation.target_guard import (
    TargetDistributionGuardConfig,
    adapt_causal_prior_to_target,
    apply_target_distribution_guard,
    compute_device_distribution,
)
from causal_smart_home.causal.generation.causal_gss import build_device_transition_graph, reweight_gss_edges
from causal_smart_home.causal.refinement.causal_tof import load_pickle_sequences
from causal_smart_home.causal_gss import load_id_name_mapping, device_key_to_name
from causal_smart_home.json_utils import jsonable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build causal-relation-guided GSS prompt artifacts.")
    parser.add_argument("--source-pkl", required=True, help="Source-context normal Gen flattened pkl.")
    parser.add_argument("--target-pkl", help="Target normal behavior; forbidden when --adaptation-mode=source_only.")
    parser.add_argument("--adaptation-mode", choices=["source_only", "target_assisted"], default="target_assisted")
    parser.add_argument("--prior-json", help="Existing causal_prior.json or resolved_causal_relation_prior.json. If absent, source-pkl is passed to existing adapter.")
    parser.add_argument("--prior-matrix-path", help="Existing causal matrix .json/.npy/.csv.")
    parser.add_argument("--adapter-mode", default="existing", choices=["existing", "compact_fallback"])
    parser.add_argument("--level", default="device", choices=["device", "action", "device_action"])
    parser.add_argument("--lag", type=int, default=4)
    parser.add_argument("--sparse-threshold", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device-dict", help="Optional JSON or dictionary.py device id/name mapping for readable hints.")
    parser.add_argument("--out-prompt", required=True)
    parser.add_argument("--out-prior-json", required=True)
    parser.add_argument("--out-target-adapted-prior", help="Defaults to target_adapted_causal_prior.json next to the prompt.")
    parser.add_argument("--out-guard-report", help="Optional historical over-use guard audit JSON.")
    parser.add_argument("--out-reweighted-hints", required=True)
    parser.add_argument("--out-config", help="Defaults to config.json next to out-prompt.")
    parser.add_argument("--lambda-causal", type=float, default=1.0)
    parser.add_argument("--reweight-mode", choices=["additive", "multiplicative"], default="multiplicative")
    parser.add_argument("--add-causal-edges", dest="add_causal_edges", action="store_true", default=True)
    parser.add_argument("--no-add-causal-edges", dest="add_causal_edges", action="store_false")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--guard-mode", choices=["suppress", "downweight"], default="downweight")
    parser.add_argument("--max-overuse-ratio", type=float, default=1.25)
    parser.add_argument("--min-target-freq", type=float, default=0.001)
    parser.add_argument("--downweight-factor", type=float, default=0.25)
    parser.add_argument("--endpoint-policy", choices=["target", "source_or_target", "both"], default="target")
    return parser.parse_args()


def main(argv: list[str] | None = None) -> None:
    args = parse_args() if argv is None else parse_args_from(argv)
    source_pkl = Path(args.source_pkl).resolve()
    target_pkl = Path(args.target_pkl).resolve() if args.target_pkl else None
    if not source_pkl.exists():
        raise FileNotFoundError(f"--source-pkl not found: {source_pkl}")
    if args.adaptation_mode == "source_only" and target_pkl is not None:
        raise ValueError("source_only causal GSS forbids --target-pkl")
    if args.adaptation_mode == "target_assisted" and target_pkl is None:
        raise ValueError("target_assisted causal GSS requires --target-pkl")
    if target_pkl is not None and not target_pkl.exists():
        raise FileNotFoundError(f"--target-pkl not found: {target_pkl}")

    out_prompt = Path(args.out_prompt).resolve()
    out_prior = Path(args.out_prior_json).resolve()
    out_adapted = (
        Path(args.out_target_adapted_prior).resolve()
        if args.out_target_adapted_prior
        else out_prompt.parent / "target_adapted_causal_prior.json"
    )
    out_guard = Path(args.out_guard_report).resolve() if args.out_guard_report else out_prompt.parent / "guard_report.json"
    out_hints = Path(args.out_reweighted_hints).resolve()
    out_config = Path(args.out_config).resolve() if args.out_config else out_prompt.parent / "config.json"
    for path in (out_prompt, out_prior, out_adapted, out_guard, out_hints, out_config):
        path.parent.mkdir(parents=True, exist_ok=True)

    prior = resolve_causal_relation_prior(
        prior_json=args.prior_json,
        prior_matrix_path=args.prior_matrix_path,
        source_pkl=str(source_pkl) if not args.prior_json and not args.prior_matrix_path else None,
        out_dir=str(out_prompt.parent),
        adapter_mode=args.adapter_mode,
        level=args.level,
        lag=args.lag,
        sparse_threshold=args.sparse_threshold,
        seed=args.seed,
    )
    prior.save(out_prior)

    # Source data discovers C and GSS transitions. Target normal data supplies
    # only P_target for the explicitly target-aware adaptation stage.
    device_mapping = load_id_name_mapping(args.device_dict, preferred_names=("sp_devices_dict", "fr_devices_dict", "us_devices_dict", "device_dict")) if args.device_dict else {}
    source_sequences = load_pickle_sequences(source_pkl)
    transition_graph = build_device_transition_graph(source_sequences, device_name_map=device_mapping)
    source_distribution = compute_device_distribution(source_sequences)
    causal_edges = annotate_edge_names(prior.top_causal_edges, device_mapping)
    target_distribution = None
    adapted_payload = None
    guard_report = None
    causal_edges_for_gss = causal_edges
    if args.adaptation_mode == "target_assisted":
        target_sequences = load_pickle_sequences(target_pkl)
        target_distribution = compute_device_distribution(target_sequences)
        adapted_edges, adapted_payload = adapt_causal_prior_to_target(
            causal_edges,
            target_distribution,
            matrix=prior.matrix,
            channels=prior.channels,
        )
        out_adapted.write_text(json.dumps(jsonable(adapted_payload), ensure_ascii=False, indent=2), encoding="utf-8")
        guard_config = TargetDistributionGuardConfig(
            max_overuse_ratio=args.max_overuse_ratio,
            min_target_freq=args.min_target_freq,
            mode=args.guard_mode,
            downweight_factor=args.downweight_factor,
            endpoint_policy=args.endpoint_policy,
        )
        causal_edges_for_gss, guard_report = apply_target_distribution_guard(
            adapted_edges,
            generated_or_prompt_distribution=source_distribution,
            target_distribution=target_distribution,
            config=guard_config,
        )
        out_guard.write_text(json.dumps(jsonable(guard_report), ensure_ascii=False, indent=2), encoding="utf-8")

    reweighted = reweight_gss_edges(
        transition_graph["edges"],
        causal_edges_for_gss,
        lambda_causal=args.lambda_causal,
        mode=args.reweight_mode,
        add_causal_edges=args.add_causal_edges,
        top_k=args.top_k,
    )
    reweighted["raw_transition_graph_summary"] = {
        "num_edges": transition_graph["num_edges"],
        "num_sequences": transition_graph["num_sequences"],
    }
    reweighted["causal_relation_source"] = prior.causal_relation_source
    reweighted["adaptation_mode"] = args.adaptation_mode
    reweighted["target_data_used"] = args.adaptation_mode == "target_assisted"
    if args.adaptation_mode == "target_assisted":
        reweighted["target_adapted_causal_prior_path"] = str(out_adapted)
        reweighted["guard_report_path"] = str(out_guard)
    out_hints.write_text(json.dumps(jsonable(reweighted), ensure_ascii=False, indent=2), encoding="utf-8")

    prompt = build_prompt_text(prior, transition_graph, adapted_payload, guard_report, reweighted, args)
    out_prompt.write_text(prompt, encoding="utf-8")

    config = {
        "script": Path(__file__).name,
        "args": vars(args),
        "source_pkl": str(source_pkl),
        "target_pkl": str(target_pkl) if target_pkl else None,
        "target_data_used": args.adaptation_mode == "target_assisted",
        "adaptation_mode": args.adaptation_mode,
        "outputs": {
            "prompt": str(out_prompt),
            "resolved_causal_relation_prior": str(out_prior),
            "causal_reweighted_gss_hints": str(out_hints),
            "config": str(out_config),
        },
        "causal_relation_source": prior.causal_relation_source,
        "source_distribution": source_distribution,
        "target_distribution": target_distribution,
        "summary": reweighted.get("summary", {}),
    }
    if adapted_payload is not None:
        config["outputs"]["target_adapted_causal_prior"] = str(out_adapted)
        config["outputs"]["guard_report"] = str(out_guard)
        config["before_after_causal_edge_statistics"] = adapted_payload["edge_statistics"]
    out_config.write_text(json.dumps(jsonable(config), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved prompt: {out_prompt}")
    print(f"saved resolved prior: {out_prior}")
    if args.adaptation_mode == "target_assisted":
        print(f"saved target-adapted prior: {out_adapted}")
        print(f"saved guard report: {out_guard}")
    print(f"saved reweighted hints: {out_hints}")
    print(f"saved config: {out_config}")


def parse_args_from(argv: list[str]) -> argparse.Namespace:
    old = sys.argv
    try:
        sys.argv = [old[0]] + argv
        return parse_args()
    finally:
        sys.argv = old


def annotate_edge_names(edges: list[dict[str, Any]], device_mapping: dict[int, str]) -> list[dict[str, Any]]:
    """给因果边字典补充人类可读的设备名。"""
    out = []
    for edge in edges:
        row = dict(edge)
        row.setdefault("source_name", device_key_to_name(str(row.get("source", "")), device_mapping))
        row.setdefault("target_name", device_key_to_name(str(row.get("target", "")), device_mapping))
        out.append(row)
    return out


def build_prompt_text(
    prior: ResolvedCausalRelationPrior,
    transition_graph: dict,
    adapted_payload: dict,
    guard_report: dict,
    reweighted: dict,
    args: argparse.Namespace,
) -> str:
    """Render target-aware causal-GSS guidance."""
    transition_edges = transition_graph.get("edges", [])[: args.top_k]
    raw_edges = prior.top_causal_edges[: args.top_k]
    causal_reweighted_edges = reweighted.get("edges", [])[: args.top_k]
    payload = {
        "original_gen_gss_transition_hints": transition_edges,
        "raw_causal_relation_hints": raw_edges,
        "target_adapted_causal_hints": adapted_payload.get("edges", [])[: args.top_k] if adapted_payload else [],
        "target_distribution_guard_summary": {
            "num_suppressed_edges": guard_report.get("num_suppressed_edges", 0) if guard_report else 0,
            "num_downweighted_edges": guard_report.get("num_downweighted_edges", 0) if guard_report else 0,
        },
        "causal_reweighted_gss_hints": causal_reweighted_edges,
    }
    return "\n".join(
        [
            "You are an IoT behavior-sequence synthesis assistant for Gen.",
            "Generate target-context smart-home behavior sequences using the structural hints below.",
            "",
            "Important guidance precedence:",
            "Use the causal-reweighted GSS hints as the primary structural guidance.",
            "Use raw causal-relation hints only as weak background evidence.",
            "If raw causal-relation hints conflict with reweighted hints, follow the reweighted hints.",
            (
                "The causal prior has been weighted by the target normal device distribution."
                if args.adaptation_mode == "target_assisted"
                else "No target behavior sample or target empirical distribution was used; follow source-derived causal hints only."
            ),
            "Do not treat causal relation edges as physical ground-truth causality; they are source-context predictive causal signals.",
            "Keep all generated behaviors in the legal Gen flattened format.",
            "",
            "JSON structural hints:",
            json.dumps(jsonable(payload), ensure_ascii=False, indent=2),
            "",
            "Return generated sequences only in the expected Gen format; do not explain unless asked.",
        ]
    )


if __name__ == "__main__":
    main()
