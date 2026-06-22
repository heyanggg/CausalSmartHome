from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from causal_smart_home.causal_tof import extract_guarded_edges, load_pickle_sequences, score_sequences_causal_tof
from causal_smart_home.target_distribution_guard import compute_device_distribution


def add_stage4a_args(parser: argparse.ArgumentParser, scenario: str) -> None:
    parser.add_argument("--source-pkl", required=True)
    parser.add_argument("--target-pkl", required=True)
    parser.add_argument("--generated-pkl", help="Optional generated sequences to evaluate; if absent, quality metrics mark generation as missing.")
    parser.add_argument("--prior-json")
    parser.add_argument("--prior-matrix-path")
    parser.add_argument("--gcad-project-dir")
    parser.add_argument("--out-dir", default=f"outputs/gcad_gss_stage4/{scenario}_guarded_reweighted_seed2024")
    parser.add_argument("--lambda-causal", type=float, default=1.0)
    parser.add_argument("--reweight-mode", choices=["additive", "multiplicative"], default="multiplicative")
    parser.add_argument("--guard-mode", choices=["suppress", "downweight"], default="suppress")
    parser.add_argument("--endpoint-policy", choices=["target", "source_or_target", "both"], default="target")
    parser.add_argument("--max-overuse-ratio", type=float, default=1.25)
    parser.add_argument("--sparse-threshold", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device-dict", help="Optional JSON or dictionary.py device mapping for readable reports.")


def run_stage4a(args: argparse.Namespace, scenario: str) -> None:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    build_script = REPO_ROOT / "scripts" / "build_guarded_causal_reweighted_gss_prompt.py"
    cmd = [
        sys.executable,
        str(build_script),
        "--source-pkl",
        str(Path(args.source_pkl).resolve()),
        "--target-pkl",
        str(Path(args.target_pkl).resolve()),
        "--out-prompt",
        str(out_dir / "prompt.txt"),
        "--out-prior-json",
        str(out_dir / "resolved_gcad_prior.json"),
        "--out-guard-report",
        str(out_dir / "guard_report.json"),
        "--out-reweighted-hints",
        str(out_dir / "guarded_reweighted_gss_hints.json"),
        "--out-config",
        str(out_dir / "config.json"),
        "--lambda-causal",
        str(args.lambda_causal),
        "--reweight-mode",
        args.reweight_mode,
        "--guard-mode",
        args.guard_mode,
        "--endpoint-policy",
        args.endpoint_policy,
        "--max-overuse-ratio",
        str(args.max_overuse_ratio),
        "--sparse-threshold",
        str(args.sparse_threshold),
        "--top-k",
        str(args.top_k),
        "--seed",
        str(args.seed),
    ]
    if args.prior_json:
        cmd.extend(["--prior-json", str(Path(args.prior_json).resolve())])
    if args.prior_matrix_path:
        cmd.extend(["--prior-matrix-path", str(Path(args.prior_matrix_path).resolve())])
    if args.gcad_project_dir:
        cmd.extend(["--gcad-project-dir", str(Path(args.gcad_project_dir).resolve())])
    if getattr(args, "device_dict", None):
        cmd.extend(["--device-dict", str(Path(args.device_dict).resolve())])
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    generated_out = out_dir / "generated.pkl"
    if args.generated_pkl:
        src = Path(args.generated_pkl).resolve()
        if not src.exists():
            raise FileNotFoundError(f"--generated-pkl not found: {src}")
        shutil.copyfile(src, generated_out)
    else:
        with open(generated_out, "wb") as f:
            pickle.dump([], f)

    metrics = compute_generation_quality(
        generated_pkl=generated_out,
        target_pkl=Path(args.target_pkl).resolve(),
        hints_json=out_dir / "guarded_reweighted_gss_hints.json",
        guard_report_json=out_dir / "guard_report.json",
        generated_missing=not bool(args.generated_pkl),
    )
    metrics["scenario"] = scenario
    metrics["seed"] = args.seed
    (out_dir / "generation_quality_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved Stage 4A outputs: {out_dir}")


def compute_generation_quality(
    generated_pkl: Path,
    target_pkl: Path,
    hints_json: Path,
    guard_report_json: Path,
    generated_missing: bool = False,
) -> dict[str, Any]:
    generated = load_pickle_sequences(generated_pkl)
    target = load_pickle_sequences(target_pkl)
    hints_payload = json.loads(hints_json.read_text(encoding="utf-8"))
    guard_report = json.loads(guard_report_json.read_text(encoding="utf-8"))
    guarded_edges = extract_guarded_edges(hints_payload)
    target_distribution = compute_device_distribution(target)
    scores = score_sequences_causal_tof(generated, guarded_edges, target_distribution=target_distribution, mode="weight") if generated else []
    return {
        "generated_missing": generated_missing,
        "generated_size": len(generated),
        "target_size": len(target),
        "low_evidence_rate": _mean([1.0 if score.get("checked_edge_weight", 0.0) == 0 else 0.0 for score in scores]),
        "causal_coverage": _mean([float(score.get("causal_coverage", 0.0)) for score in scores]),
        "causal_violation_rate": _mean([float(score.get("causal_violation", 0.0)) for score in scores]),
        "action_js_to_target": js_for_level(generated, target, "action") if generated else None,
        "device_js_to_target": js_for_level(generated, target, "device") if generated else None,
        "transition_js_to_target": transition_js(generated, target) if generated else None,
        "tof_kept_rate": 1.0 if generated else 0.0,
        "guarded_edge_count": len([edge for edge in guarded_edges if float(edge.get("guarded_causal_strength", edge.get("guarded_weight", edge.get("weight", 0.0)))) > 0]),
        "suppressed_edge_count": guard_report.get("num_suppressed_edges", 0),
        "downweighted_edge_count": guard_report.get("num_downweighted_edges", 0),
        "avg_guarded_causal_strength": _mean([
            float(edge.get("guarded_causal_strength", edge.get("guarded_weight", edge.get("weight", 0.0)))) for edge in guarded_edges
        ]),
    }


def add_stage4b_args(parser: argparse.ArgumentParser, scenario: str, weighted: bool = False) -> None:
    parser.add_argument("--stage4a-dir", help="Stage 4A output directory used as provenance for this AD registration/dry-run.")
    if weighted:
        parser.add_argument("--weighted-generated-pkl", help="Causal-TOF weighted/resampled generated pkl used for downstream AD.")
    parser.add_argument("--out-dir", default=f"outputs/gcad_gss_stage4/{scenario}_{'causal_tof_weighted' if weighted else 'guarded_reweighted'}_ad_seed2024")
    parser.add_argument("--metrics-json", help="Real downstream AD metrics JSON to register as Stage 4B output.")
    parser.add_argument("--baseline-metrics-json", help="Original prompt baseline metrics JSON for deltas.")
    parser.add_argument("--dry-run", action="store_true", help="Write a transparent placeholder when real SmartGuard data is unavailable.")
    parser.add_argument("--seed", type=int, default=2024)


def run_stage4b(args: argparse.Namespace, scenario: str, weighted: bool = False) -> None:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stage4a_dir = Path(args.stage4a_dir).resolve() if getattr(args, "stage4a_dir", None) else None
    if stage4a_dir and not stage4a_dir.exists():
        raise FileNotFoundError(f"--stage4a-dir not found: {stage4a_dir}")
    weighted_generated_pkl = Path(args.weighted_generated_pkl).resolve() if weighted and getattr(args, "weighted_generated_pkl", None) else None
    if weighted_generated_pkl and not weighted_generated_pkl.exists():
        raise FileNotFoundError(f"--weighted-generated-pkl not found: {weighted_generated_pkl}")
    if not args.metrics_json and not args.dry_run:
        raise ValueError("Stage 4B requires --metrics-json from a real downstream AD run or --dry-run to record missing data explicitly")
    if args.metrics_json:
        metrics_path = Path(args.metrics_json).resolve()
        if not metrics_path.exists():
            raise FileNotFoundError(f"--metrics-json not found: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = normalize_ad_metrics(metrics)
    else:
        metrics = {
            "precision": None,
            "recall": None,
            "f1": None,
            "fpr": None,
            "accuracy": None,
            "dry_run": True,
            "missing_reason": "Real SmartGuard downstream AD data/metrics were not provided in this environment.",
        }
    baseline = None
    if args.baseline_metrics_json:
        base_path = Path(args.baseline_metrics_json).resolve()
        if not base_path.exists():
            raise FileNotFoundError(f"--baseline-metrics-json not found: {base_path}")
        baseline = normalize_ad_metrics(json.loads(base_path.read_text(encoding="utf-8")))
    metrics["delta_f1_vs_original_prompt"] = _delta(metrics.get("f1"), baseline.get("f1") if baseline else None)
    metrics["delta_fpr_vs_original_prompt"] = _delta(metrics.get("fpr"), baseline.get("fpr") if baseline else None)
    metrics["scenario"] = scenario
    metrics["seed"] = args.seed
    metrics["weighted"] = weighted
    metrics["stage4a_dir"] = str(stage4a_dir) if stage4a_dir else None
    metrics["weighted_generated_pkl"] = str(weighted_generated_pkl) if weighted_generated_pkl else None
    (out_dir / "downstream_ad_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved Stage 4B metrics: {out_dir / 'downstream_ad_metrics.json'}")


def normalize_ad_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    aliases = {"f1_score": "f1", "F1": "f1", "false_positive_rate": "fpr"}
    out = dict(metrics)
    for src, dst in aliases.items():
        if src in out and dst not in out:
            out[dst] = out[src]
    for key in ("precision", "recall", "f1", "fpr", "accuracy"):
        out.setdefault(key, None)
    return out


def js_for_level(a: Sequence, b: Sequence, level: str) -> float:
    return jensen_shannon(_distribution_for_level(a, level), _distribution_for_level(b, level))


def transition_js(a: Sequence, b: Sequence) -> float:
    return jensen_shannon(_transition_distribution(a), _transition_distribution(b))


def _distribution_for_level(sequences: Sequence, level: str) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for seq in sequences:
        for event in seq:
            counts[event.key(level)] += 1
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()} if total else {}


def _transition_distribution(sequences: Sequence) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for seq in sequences:
        keys = [event.key("device") for event in seq]
        for src, tgt in zip(keys, keys[1:]):
            counts[f"{src}->{tgt}"] += 1
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()} if total else {}


def jensen_shannon(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    pp = {key: float(p.get(key, 0.0)) for key in keys}
    qq = {key: float(q.get(key, 0.0)) for key in keys}
    mp = {key: 0.5 * (pp[key] + qq[key]) for key in keys}
    return math.sqrt(0.5 * _kl(pp, mp) + 0.5 * _kl(qq, mp))


def _kl(p: dict[str, float], q: dict[str, float]) -> float:
    total = 0.0
    for key, value in p.items():
        if value > 0 and q.get(key, 0.0) > 0:
            total += value * math.log(value / q[key], 2)
    return total


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _delta(value: Any, baseline: Any) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value) - float(baseline)
