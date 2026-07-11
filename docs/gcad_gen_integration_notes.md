# Source-only causal relation integration with SmartGen

## Method boundary

The current method adds a source-derived causal relation prior to SmartGen's
Graph-guided Sequence Synthesis. It does not use a target empirical
distribution and does not add a post-Gen Causal-TOF stage.

```text
source normal sequences
  -> EventTensorizer
  -> causal relation prior
  -> source GSS transition graph
  -> causal-reweighted GSS hints
  -> Codex context adaptation
  -> Gen original TOF
  -> downstream evaluation
```

Target normal and attack behavior are evaluation-only.

## Causal prior

Gen events use `[day, hour_slot, device_id, action_id]`. `EventTensorizer`
projects source events into device-level time channels such as `d:13`. The
causal adapter produces directed predictive strengths between these channels.
These strengths are soft structural evidence, not physical causal truth.

The prior resolver accepts an existing prior, a matrix, or the local source-only
fallback miner. Every route produces a normalized list of device edges.

## GSS fusion

The original GSS transition score is:

```text
transition_score(A -> B) = count(A followed by B) / outgoing_count(A)
```

For a causal edge strength `c`, the current multiplicative fusion is:

```text
normalized_transition = transition_score / max_transition
normalized_causal = c / max_causal
final_score = normalized_transition * (1 + lambda_causal * normalized_causal)
```

With `add_causal_edges=True`, a source-derived causal relation absent from
adjacent GSS transitions may also be emitted as an augmented hint. No target
frequency changes these scores.

## Generation and filtering

Codex receives source sequences/hints, legal device/action information, and the
declared context transition. It must record `target_data_used=false`.

The generated JSONL is validated for flat-quadruple structure and dictionary
legality. SmartGen's original two-stage TOF is then the only synthetic-data
filter. The downstream evaluator loads target data only after generation and
TOF have completed.

## Removed design

The earlier design used target `test.pkl` or `split_test.pkl` in Target
Distribution Guard and Causal-TOF. That changed the task into target-reference-
assisted adaptation and risked test-distribution leakage. It was removed from
the current tree on 2026-07-11 and remains recoverable from:

```text
archive-before-target-guard-redesign-20260711
```
