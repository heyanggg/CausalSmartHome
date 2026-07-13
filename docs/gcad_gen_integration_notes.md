# Target-aware causal relation integration with SmartGen

## Method boundary

The current method adapts a source-derived causal relation prior with the target
normal event distribution before SmartGen's Graph-guided Sequence Synthesis,
then applies a continuous causal-consistency stage after Gen original TOF.

```text
source normal sequences
  -> EventTensorizer
  -> causal relation prior
  -> C(i,j) * P_target(i) * P_target(j)
  -> source GSS transition graph
  -> causal-reweighted GSS hints
  -> Codex context adaptation
  -> Gen original TOF
  -> Causal-TOF consistency score
  -> downstream evaluation
```

Target normal behavior is used only for declared causal adaptation, distribution
penalty, and evaluation. Attack behavior remains evaluation-only.

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
frequency changes the original transition counts; target probabilities only
adapt causal strengths before fusion.

## Generation and filtering

Codex receives source sequences/hints, legal device/action information, the
declared context transition, and target-adapted causal artifacts. It records
`target_data_used=true` with the target data role.

The generated JSONL is validated for flat-quadruple structure and dictionary
legality. SmartGen's original two-stage TOF is unchanged. Causal-TOF then uses
continuous mean causal strength rather than binary violation counting.

## Historical design

The zero-target redesign from 2026-07-11 remains recoverable from:

```text
archive-before-target-guard-redesign-20260711
```
