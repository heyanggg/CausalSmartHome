from __future__ import annotations

import json
from typing import Sequence, Any
from .causal_prior import CausalPrior
from .schema import BehaviorSequence


def prior_to_json_hints(prior: CausalPrior, top_k: int = 20) -> dict[str, Any]:
    edges = prior.top_edges(k=top_k, include_self=False)
    return {
        "hint_type": "subsequence_granger_causality",
        "lag": prior.lag,
        "interpretation": "source behavior/action tends to Granger-cause or precede target behavior/action in normal routines; preserve these causal orders unless the new context explicitly changes the involved device.",
        "top_causal_edges": edges,
    }


def format_numeric_sequences_for_prompt(sequences: Sequence[BehaviorSequence], max_sequences: int = 20) -> list[list[int]]:
    return [seq.to_flat_numeric() for seq in sequences[:max_sequences]]


def build_causal_smartgen_prompt(
    original_sequences: Sequence[BehaviorSequence],
    prior: CausalPrior,
    device_information: dict[str, Any] | str,
    original_context: str,
    new_context: str,
    transition_hints: dict[str, Any] | None = None,
    max_sequences: int = 20,
    top_k_edges: int = 20,
) -> str:
    """Build a SmartGen-compatible prompt with additional causal JSON hints.

    This does not alter SmartGen. It only changes the prompt payload supplied to
    the LLM stage, so it is a pure glue layer.
    """
    causal_hints = prior_to_json_hints(prior, top_k=top_k_edges)
    payload = {
        "original_context": original_context,
        "new_context": new_context,
        "device_information": device_information,
        "compressed_original_sequences": format_numeric_sequences_for_prompt(original_sequences, max_sequences=max_sequences),
        "smartgen_transition_hints": transition_hints or {},
        "causal_hints": causal_hints,
    }
    return (
        "You are an IoT expert. Generate smart-home behavior sequences for the new context.\n"
        "Follow the original SmartGen output convention: return only <seq [[...], [...]] seq>.\n"
        "Requirements:\n"
        "1. Adapt behaviors to the new context.\n"
        "2. Do not invent devices or actions outside device_information.\n"
        "3. Preserve routine-independent behavior patterns from compressed_original_sequences.\n"
        "4. Use smartgen_transition_hints to preserve high-frequency transitions.\n"
        "5. Use causal_hints as higher-order constraints: if both source and target behaviors appear, keep their normal causal/predecessor order where reasonable.\n"
        "6. Avoid sequences that satisfy only local transitions but violate many causal_hints.\n"
        "JSON_PAYLOAD:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
