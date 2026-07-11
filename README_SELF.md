# CausalSmartHome internal experiment policy

## Formal setting

The current method is strict zero-target-data adaptation. Before final
evaluation, no code may load target-context behavior samples or empirical
target distributions.

Allowed generation inputs:

- source-context normal behavior;
- source-derived causal relation prior and GSS transitions;
- device/action legality dictionaries;
- declared transition (`winter -> spring`, `daytime -> night`, or
  `single -> multiple`);
- general semantic reasoning about that declared transition.

Forbidden generation inputs:

- target `trn.pkl`, `vld.pkl`, `rs_vld.pkl`, `test.pkl`, or `split_test.pkl`;
- downstream target-normal or attack files;
- metrics or false-positive samples from the final evaluation.

Formal pipeline and variant:

```text
source-only causal prior -> causal-reweighted GSS -> Codex generation
-> Gen original TOF -> Gen downstream AD

proposed_zero_target_causal_gss_codex
```

Target Distribution Guard and Causal-TOF are not part of the current method.
Their code and experiments are available only through the rollback tag/snapshot.

## Evaluation policy

- Run each declared seed independently and report it independently.
- Target evaluation data may be opened only inside downstream AD evaluation.
- Generation changes must be justified by source-only validation, legality,
  duplicate/coverage reports, or Gen TOF diagnostics—not by final test errors.
- If the final test was already observed, subsequent runs must be labeled
  exploratory rather than untouched-test results.
- Do not hide unsuccessful exploratory runs.
- Do not overwrite `data/main_experiment/`; write new runs under
  `outputs/zero_target_runs/`.

## Historical results

Artifacts under `data/main_experiment/` used target reference distributions and
the removed Causal-TOF stage. They are useful for provenance but cannot be
reported as zero-target-data results.

Pre-redesign rollback:

```text
commit: 1d4f7734b1953eebf1e16976c4dae3eb9a81aa0b
tag: archive-before-target-guard-redesign-20260711
snapshot: /home/heyang/projects/CausalSmartHome_checkpoints/20260711_before_target_guard_redesign/
```
