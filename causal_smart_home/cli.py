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
from .demo_data import make_toy_normal_sequences, make_toy_generated_candidates


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
