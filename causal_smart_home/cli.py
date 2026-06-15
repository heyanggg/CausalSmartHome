from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from .schema import load_numeric_sequences, dump_numeric_sequences
from .smartgen_adapter import SmartGenAdapter
from .pipeline import CausalSmartHomePipeline
from .causal_prior import CausalPrior
from .causal_prompt import build_causal_smartgen_prompt
from .causal_filter import CausalConsistencyFilter
from .filter_sweep import build_filter_sweep_configs, run_filter_sweep
from .smartguard_experiment import (
    SmartGuardRunConfig,
    default_smartguard_paths,
    run_smartguard_experiment,
    run_smartguard_sweep,
)
from .demo_data import make_toy_normal_sequences, make_toy_generated_candidates


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_optional_float_list(value: str | None) -> list[float | None] | None:
    if value is None:
        return None
    out: list[float | None] = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if item.lower() in {"none", "null"}:
            out.append(None)
        else:
            out.append(float(item))
    return out


def _load_pkl(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pkl(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def cmd_build_prior(args) -> None:
    raw = _load_pkl(args.train_pkl)
    sequences = load_numeric_sequences(raw)
    pipeline = CausalSmartHomePipeline(args.out_dir)
    prior = pipeline.build_prior(
        sequences,
        lag=args.lag,
        epochs=args.epochs,
        sparse_threshold=args.sparse_threshold,
        level=args.level,
    )
    out = Path(args.out_dir) / "causal_prior.json"
    prior.save(out)
    print(f"saved {out}")


def cmd_prompt(args) -> None:
    prior = CausalPrior.load(args.prior_json)
    raw = _load_pkl(args.compressed_pkl)
    sequences = load_numeric_sequences(raw)
    with open(args.device_info_json, "r", encoding="utf-8") as f:
        device_info = json.load(f)
    transition_hints = None
    if args.transition_json:
        with open(args.transition_json, "r", encoding="utf-8") as f:
            transition_hints = json.load(f)
    prompt = build_causal_smartgen_prompt(
        sequences,
        prior,
        device_information=device_info,
        original_context=args.original_context,
        new_context=args.new_context,
        transition_hints=transition_hints or SmartGenAdapter.build_transition_hints(sequences),
        max_sequences=args.max_sequences,
        top_k_edges=args.top_k_edges,
    )
    out = Path(args.out_prompt)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(prompt, encoding="utf-8")
    print(f"saved {out}")


def cmd_filter(args) -> None:
    prior = CausalPrior.load(args.prior_json)
    raw = _load_pkl(args.generated_pkl)
    sequences = load_numeric_sequences(raw)
    result = CausalConsistencyFilter(prior, top_k_edges=args.top_k_edges).filter(sequences, min_coverage=args.min_coverage)
    kept_raw = dump_numeric_sequences(result.kept)
    _save_pkl(args.out_pkl, kept_raw)
    scores_path = Path(args.out_scores)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(result.scores, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"kept={len(result.kept)} rejected={len(result.rejected)}")
    print(f"saved {args.out_pkl}")
    print(f"saved {scores_path}")


def cmd_sweep_filter(args) -> None:
    prior = CausalPrior.load(args.prior_json)
    raw = _load_pkl(args.generated_pkl)
    sequences = load_numeric_sequences(raw)
    configs = build_filter_sweep_configs(
        top_k_edges=_parse_int_list(args.top_k_edges),
        min_coverages=_parse_float_list(args.min_coverages),
        min_checked_edges=_parse_int_list(args.min_checked_edges),
        min_edge_weights=_parse_optional_float_list(args.min_edge_weights),
    )
    tag = args.tag or Path(args.generated_pkl).stem
    rows = run_filter_sweep(
        prior,
        sequences,
        configs,
        out_dir=args.out_dir,
        tag=tag,
        write_kept=not args.summary_only,
        write_scores=args.write_scores,
        summary_prefix=args.summary_prefix,
        sequence_length=args.sequence_length,
        pad_value=args.pad_value,
    )
    csv_path = Path(args.out_dir) / f"{args.summary_prefix}_summary.csv"
    json_path = Path(args.out_dir) / f"{args.summary_prefix}_summary.json"
    print(f"scanned={len(rows)} raw={len(sequences)}")
    print(f"saved {csv_path}")
    print(f"saved {json_path}")


def _smartguard_config_from_args(args, add_pkl: str | Path, tag: str) -> SmartGuardRunConfig:
    smartguard_root = Path(args.smartguard_root).resolve()
    defaults = default_smartguard_paths(smartguard_root, args.dataset)
    return SmartGuardRunConfig(
        smartguard_root=smartguard_root,
        dataset=args.dataset,
        base_train_pkl=Path(args.base_train_pkl or defaults["base_train_pkl"]).resolve(),
        add_pkl=Path(add_pkl).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        tag=tag,
        vld_pkl=Path(args.vld_pkl).resolve() if args.vld_pkl else defaults["vld_pkl"],
        test_pkl=Path(args.test_pkl).resolve() if args.test_pkl else defaults["test_pkl"],
        sequence_length=args.sequence_length,
        pad_value=args.pad_value,
        epochs=args.epochs,
        threshold_percentage=args.threshold_percentage,
        model=args.model,
        mask_strategy=args.mask_strategy,
        mask_ratio=args.mask_ratio,
        mask_step=args.mask_step,
        layer=args.layer,
        batch=args.batch,
        embedding=args.embedding,
        TTPE=args.TTPE,
        LDMS=args.LDMS,
        seed=args.seed,
        attacks=tuple(_parse_str_list(args.attacks)),
        dry_run=args.dry_run,
    )


def cmd_smartguard_eval(args) -> None:
    tag = args.tag or Path(args.add_pkl).stem
    config = _smartguard_config_from_args(args, args.add_pkl, tag)
    payload = run_smartguard_experiment(config)
    print(f"saved {payload['result_path']}")
    aggregate = payload.get("aggregate")
    if aggregate:
        print(
            "aggregate "
            f"recall={aggregate['recall']:.4f} "
            f"precision={aggregate['precision']:.4f} "
            f"f1={aggregate['f1_score']:.4f}"
        )


def cmd_smartguard_sweep_eval(args) -> None:
    slugs = _parse_str_list(args.select_slugs)
    base_config = _smartguard_config_from_args(args, args.sweep_summary, args.tag)
    rows = run_smartguard_sweep(args.sweep_summary, base_config, slugs=slugs)
    out = Path(args.out_dir)
    print(f"evaluated={len(rows)}")
    print(f"saved {out / 'smartguard_sweep_eval_summary.csv'}")
    print(f"saved {out / 'smartguard_sweep_eval_summary.json'}")


def cmd_demo(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    normal = make_toy_normal_sequences(n=args.num_sequences)
    candidates = make_toy_generated_candidates()
    _save_pkl(out_dir / "toy_normal.pkl", dump_numeric_sequences(normal))
    _save_pkl(out_dir / "toy_generated_candidates.pkl", dump_numeric_sequences(candidates))

    pipeline = CausalSmartHomePipeline(out_dir)
    prior = pipeline.build_prior(normal, lag=args.lag, epochs=args.epochs, sparse_threshold=args.sparse_threshold)
    prior_path = out_dir / "causal_prior.json"
    prior.save(prior_path)

    device_info = {"toy_actions": {"10": "arrive", "11": "unlock", "12": "light_on", "13": "curtain_close", "14": "cook", "15": "eat", "16": "wash_dishes"}}
    prompt = pipeline.build_prompt(normal[:10], prior, device_info, "daytime single-person routine", "night shift routine")
    prompt_path = out_dir / "causal_smartgen_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    result = pipeline.filter_generated(candidates, prior, min_coverage=args.min_coverage)
    _save_pkl(out_dir / "toy_generated_kept.pkl", dump_numeric_sequences(result.kept))
    (out_dir / "causal_filter_scores.json").write_text(json.dumps(result.scores, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "normal_sequences": len(normal),
        "candidate_sequences": len(candidates),
        "kept_sequences": len(result.kept),
        "rejected_sequences": len(result.rejected),
        "top_edges": prior.top_edges(k=10),
        "last_train_loss": prior.meta.get("train_loss_last") if prior.meta else None,
    }
    (out_dir / "demo_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Causal Smart Home CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-prior", help="mine GCAD-style causal prior from SmartGuard/SmartGen numeric pkl")
    p.add_argument("--train-pkl", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--lag", type=int, default=4)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--sparse-threshold", type=float, default=0.0)
    p.add_argument("--level", choices=["action", "device", "device_action"], default="action")
    p.set_defaults(func=cmd_build_prior)

    p = sub.add_parser("prompt", help="build SmartGen prompt with causal hints")
    p.add_argument("--prior-json", required=True)
    p.add_argument("--compressed-pkl", required=True)
    p.add_argument("--device-info-json", required=True)
    p.add_argument("--original-context", required=True)
    p.add_argument("--new-context", required=True)
    p.add_argument("--out-prompt", required=True)
    p.add_argument("--transition-json")
    p.add_argument("--max-sequences", type=int, default=20)
    p.add_argument("--top-k-edges", type=int, default=20)
    p.set_defaults(func=cmd_prompt)

    p = sub.add_parser("filter", help="causal post-filter generated pkl sequences")
    p.add_argument("--prior-json", required=True)
    p.add_argument("--generated-pkl", required=True)
    p.add_argument("--out-pkl", required=True)
    p.add_argument("--out-scores", required=True)
    p.add_argument("--min-coverage", type=float, default=0.5)
    p.add_argument("--top-k-edges", type=int, default=30)
    p.set_defaults(func=cmd_filter)

    p = sub.add_parser("sweep-filter", help="sweep causal filter thresholds and write kept pkl candidates")
    p.add_argument("--prior-json", required=True)
    p.add_argument("--generated-pkl", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--top-k-edges", default="10,20,30")
    p.add_argument("--min-coverages", default="0.3,0.5,0.7")
    p.add_argument("--min-checked-edges", default="0,1,2,3")
    p.add_argument("--min-edge-weights")
    p.add_argument("--tag")
    p.add_argument("--summary-prefix", default="filter_sweep")
    p.add_argument("--sequence-length", type=int)
    p.add_argument("--pad-value", type=int, default=0)
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--write-scores", action="store_true")
    p.set_defaults(func=cmd_sweep_filter)

    def add_smartguard_common_options(p) -> None:
        p.add_argument("--smartguard-root", default="/home/heyang/projects/SmartGuard")
        p.add_argument("--dataset", default="fr")
        p.add_argument("--base-train-pkl")
        p.add_argument("--vld-pkl")
        p.add_argument("--test-pkl")
        p.add_argument("--out-dir", required=True)
        p.add_argument("--sequence-length", type=int, default=40)
        p.add_argument("--pad-value", type=int, default=0)
        p.add_argument("--epochs", type=int, default=60)
        p.add_argument("--threshold-percentage", type=float, default=95.0)
        p.add_argument("--model", default="SmartGuard")
        p.add_argument("--mask-strategy", default="loss_guided")
        p.add_argument("--mask-ratio", type=float, default=0.2)
        p.add_argument("--mask-step", type=int, default=4)
        p.add_argument("--layer", type=int, default=2)
        p.add_argument("--batch", type=int, default=1024)
        p.add_argument("--embedding", type=int, default=256)
        p.add_argument("--TTPE", dest="TTPE", action="store_true", default=True)
        p.add_argument("--no-TTPE", dest="TTPE", action="store_false")
        p.add_argument("--LDMS", dest="LDMS", action="store_true", default=True)
        p.add_argument("--no-LDMS", dest="LDMS", action="store_false")
        p.add_argument("--seed", type=int, default=2023)
        p.add_argument("--attacks", default="SD,MD,DM,DD")
        p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("smartguard-eval", help="train/evaluate SmartGuard with one added synthetic pkl")
    add_smartguard_common_options(p)
    p.add_argument("--add-pkl", required=True)
    p.add_argument("--tag")
    p.set_defaults(func=cmd_smartguard_eval)

    p = sub.add_parser("smartguard-sweep-eval", help="train/evaluate SmartGuard for selected filter sweep rows")
    add_smartguard_common_options(p)
    p.add_argument("--sweep-summary", required=True)
    p.add_argument("--select-slugs", default="k30_cov0p5_chk1,k30_cov0p5_chk2,k30_cov0p5_chk3")
    p.add_argument("--tag", default="causal_filter")
    p.set_defaults(func=cmd_smartguard_sweep_eval)

    p = sub.add_parser("demo", help="run toy end-to-end demo")
    p.add_argument("--out-dir", default="outputs/demo")
    p.add_argument("--num-sequences", type=int, default=80)
    p.add_argument("--lag", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--sparse-threshold", type=float, default=0.0)
    p.add_argument("--min-coverage", type=float, default=0.3)
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
