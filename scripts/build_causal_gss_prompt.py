#!/usr/bin/env python
"""构建生成阶段使用的 GCAD + Gen GSS prompt 产物。

输入：
    源上下文 normal Gen pkl、目标上下文 normal Gen pkl，以及可选的 causal prior。

输出：
    resolved prior JSON、target distribution guard report、guarded reweighted
    GSS hints，以及供 Codex/人工生成使用的 prompt 文本。
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
from causal_smart_home.causal_gss_reweight import build_device_transition_graph, reweight_gss_edges
from causal_smart_home.target_distribution_guard import (
    TargetDistributionGuardConfig,
    apply_target_distribution_guard,
    compute_device_distribution,
)
from causal_smart_home.causal_tof import load_pickle_sequences
from causal_smart_home.causal_gss import load_id_name_mapping, device_key_to_name
from causal_smart_home.json_utils import jsonable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build causal-relation-guided GSS prompt artifacts.")
    parser.add_argument("--source-pkl", required=True, help="Source-context normal Gen flattened pkl.")
    parser.add_argument("--target-pkl", required=True, help="Target-context normal pkl for distribution guard.")
    parser.add_argument("--prior-json", help="Existing causal_prior.json or resolved_causal_relation_prior.json. If absent, source-pkl is passed to existing adapter.")
    parser.add_argument("--prior-matrix-path", help="Existing causal matrix .json/.npy/.csv.")
    parser.add_argument("--adapter-mode", default="existing", choices=["existing", "compact_fallback"])
    parser.add_argument("--level", default="device", choices=["device", "action", "device_action"])
    parser.add_argument("--lag", type=int, default=4)
    parser.add_argument("--sparse-threshold", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device-dict", help="Optional JSON or dictionary.py device id/name mapping for readable guard reports.")
    parser.add_argument("--out-prompt", required=True)
    parser.add_argument("--out-prior-json", required=True)
    parser.add_argument("--out-guard-report", required=True)
    parser.add_argument("--out-reweighted-hints", required=True)
    parser.add_argument("--out-config", help="Defaults to config.json next to out-prompt.")
    parser.add_argument("--lambda-causal", type=float, default=1.0)
    parser.add_argument("--reweight-mode", choices=["additive", "multiplicative"], default="multiplicative")
    parser.add_argument("--add-causal-edges", dest="add_causal_edges", action="store_true", default=True)
    parser.add_argument("--no-add-causal-edges", dest="add_causal_edges", action="store_false")
    parser.add_argument("--guard-mode", choices=["suppress", "downweight"], default="downweight")
    parser.add_argument("--max-overuse-ratio", type=float, default=1.25)
    parser.add_argument("--min-target-freq", type=float, default=0.001)
    parser.add_argument("--downweight-factor", type=float, default=0.25)
    parser.add_argument("--endpoint-policy", choices=["target", "source_or_target", "both"], default="target")
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main(argv: list[str] | None = None) -> None:
    args = parse_args() if argv is None else parse_args_from(argv)
    source_pkl = Path(args.source_pkl).resolve()
    target_pkl = Path(args.target_pkl).resolve()
    if not source_pkl.exists():
        raise FileNotFoundError(f"--source-pkl not found: {source_pkl}")
    if not target_pkl.exists():
        raise FileNotFoundError(f"--target-pkl not found: {target_pkl}")

    out_prompt = Path(args.out_prompt).resolve()
    out_prior = Path(args.out_prior_json).resolve()
    out_guard = Path(args.out_guard_report).resolve()
    out_hints = Path(args.out_reweighted_hints).resolve()
    out_config = Path(args.out_config).resolve() if args.out_config else out_prompt.parent / "config.json"
    for path in (out_prompt, out_prior, out_guard, out_hints, out_config):
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

    # 源序列有两重用途：一是统计 Gen 原始 GSS 转移，二是在没有外部 prior
    # 时作为 GCAD prior 的挖掘来源。目标序列用于计算目标设备分布，在源
    # 上下文因果边真正影响目标生成之前先做分布保护。
    device_mapping = load_id_name_mapping(args.device_dict, preferred_names=("sp_devices_dict", "fr_devices_dict", "us_devices_dict", "device_dict")) if args.device_dict else {}
    source_sequences = load_pickle_sequences(source_pkl)
    target_sequences = load_pickle_sequences(target_pkl)
    transition_graph = build_device_transition_graph(source_sequences, device_name_map=device_mapping)
    source_distribution = compute_device_distribution(source_sequences)
    target_distribution = compute_device_distribution(target_sequences)

    guard_config = TargetDistributionGuardConfig(
        max_overuse_ratio=args.max_overuse_ratio,
        min_target_freq=args.min_target_freq,
        mode=args.guard_mode,
        downweight_factor=args.downweight_factor,
        endpoint_policy=args.endpoint_policy,
    )
    guarded_edges, guard_report = apply_target_distribution_guard(
        annotate_edge_names(prior.top_causal_edges, device_mapping),
        generated_or_prompt_distribution=source_distribution,
        target_distribution=target_distribution,
        config=guard_config,
    )
    guard_report = annotate_guard_report_device_names(guard_report, device_mapping)
    out_guard.write_text(json.dumps(jsonable(guard_report), ensure_ascii=False, indent=2), encoding="utf-8")

    reweighted = reweight_gss_edges(
        transition_graph["edges"],
        guarded_edges,
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
    reweighted["guard_report_path"] = str(out_guard)
    out_hints.write_text(json.dumps(jsonable(reweighted), ensure_ascii=False, indent=2), encoding="utf-8")

    prompt = build_prompt_text(prior, transition_graph, guard_report, reweighted, args)
    out_prompt.write_text(prompt, encoding="utf-8")

    config = {
        "script": Path(__file__).name,
        "args": vars(args),
        "source_pkl": str(source_pkl),
        "target_pkl": str(target_pkl),
        "outputs": {
            "prompt": str(out_prompt),
            "resolved_causal_relation_prior": str(out_prior),
            "guard_report": str(out_guard),
            "guarded_reweighted_gss_hints": str(out_hints),
            "config": str(out_config),
        },
        "causal_relation_source": prior.causal_relation_source,
        "source_distribution": source_distribution,
        "target_distribution": target_distribution,
        "summary": reweighted.get("summary", {}),
    }
    out_config.write_text(json.dumps(jsonable(config), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved prompt: {out_prompt}")
    print(f"saved resolved prior: {out_prior}")
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


def annotate_guard_report_device_names(report: dict[str, Any], device_mapping: dict[int, str]) -> dict[str, Any]:
    """在存在设备字典时，让 overuse report 更容易人工检查。"""
    if not device_mapping:
        return report
    out = dict(report)
    overused = []
    for item in report.get("overused_devices", []):
        row = dict(item)
        row["device_name"] = device_key_to_name(str(row.get("device_key", "")), device_mapping)
        overused.append(row)
    out["overused_devices"] = overused
    return out


def build_prompt_text(prior: ResolvedCausalRelationPrior, transition_graph: dict, guard_report: dict, reweighted: dict, args: argparse.Namespace) -> str:
    """渲染包含原始 GSS、raw GCAD 和 guarded hints 的最终 prompt。"""
    transition_edges = transition_graph.get("edges", [])[: args.top_k]
    raw_edges = prior.top_causal_edges[: args.top_k]
    guarded_edges = reweighted.get("edges", [])[: args.top_k]
    payload = {
        "original_gen_gss_transition_hints": transition_edges,
        "raw_causal_relation_hints": raw_edges,
        "target_distribution_guard_report_summary": {
            "overused_devices": guard_report.get("overused_devices", []),
            "num_suppressed_edges": guard_report.get("num_suppressed_edges", 0),
            "num_downweighted_edges": guard_report.get("num_downweighted_edges", 0),
        },
        "guarded_causal_reweighted_gss_hints": guarded_edges,
    }
    return "\n".join(
        [
            "You are an IoT behavior-sequence synthesis assistant for Gen.",
            "Generate target-context smart-home behavior sequences using the structural hints below.",
            "",
            "Important guidance precedence:",
            "Use the guarded causal-reweighted GSS hints as the primary structural guidance.",
            "Use raw causal-relation hints only as weak background evidence.",
            "If raw causal-relation hints conflict with guarded reweighted hints, follow the guarded reweighted hints.",
            "Do not over-generate devices marked as overused in the target-distribution guard report.",
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
