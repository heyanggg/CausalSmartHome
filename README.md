# CausalSmartHome

## Scope

CausalSmartHome is a zero-target-data extension of SmartGen. The method adapts
from an original smart-home context to a declared new context without reading
behavior samples, empirical distributions, or labels from that target context.

The formal pipeline is:

```text
source-context normal behavior
-> source-only causal relation prior
-> causal-reweighted GSS
-> Codex context adaptation
-> Gen original two-stage TOF
-> Gen built-in downstream AD
-> per-seed results
```

Target-context normal and attack data are loaded only by the final downstream
evaluation. They must not be supplied to prompt construction, generation, or
TOF.

The formal variant is:

```text
proposed_zero_target_causal_gss_codex
```

Target Distribution Guard and Causal-TOF were removed from the formal method on
2026-07-11 because both changed the problem into target-reference-assisted
adaptation and Causal-TOF duplicated Gen's filtering role. Historical results
remain under `data/main_experiment/` and in the rollback snapshot; they are not
zero-target-data results.

## Context transitions

| scenario | transition |
| --- | --- |
| `st` / spring | winter -> spring |
| `tt` / night | daytime -> nighttime |
| `nt` / multiple | single -> multiple occupancy |

The method may use the declared transition, source behavior, device/action
dictionary, and general context semantics. It may not inspect the target
`trn.pkl`, `vld.pkl`, `test.pkl`, `split_test.pkl`, or downstream evaluation
files during generation.

## Main commands

Run commands from the repository root in an environment with the dependencies
from `requirements.txt`.

Prepare a source-only causal-GSS prompt package:

```bash
python scripts/main_prepare_generation.py \
  --dataset fr --scenario tt --seed 2024 \
  --out-root outputs/zero_target_runs
```

Validate and package Codex-authored JSONL:

```bash
python scripts/validate_and_pack_codex_generation.py \
  --input-jsonl outputs/zero_target_runs/fr_tt/seed2024/codex_generation/generated_codex.jsonl \
  --out-pkl outputs/zero_target_runs/fr_tt/seed2024/codex_generation/generated_codex.pkl \
  --out-validation-report outputs/zero_target_runs/fr_tt/seed2024/codex_generation/validation_report.json \
  --out-generation-report outputs/zero_target_runs/fr_tt/seed2024/codex_generation/generation_report.json \
  --dictionary-py data/gen/dictionary.py \
  --dataset fr --scenario tt --scenario-key fr_tt --seed 2024 \
  --expected-count 300 \
  --source-pkl data/gen/fr/daytime/trn.pkl \
  --schema-json outputs/zero_target_runs/fr_tt/seed2024/codex_generation_package/generation_schema.json \
  --causal-hints-json outputs/zero_target_runs/fr_tt/seed2024/codex_generation_package/causal_reweighted_gss_hints.json \
  --resolved-causal-relation-prior-json outputs/zero_target_runs/fr_tt/seed2024/codex_generation_package/resolved_causal_relation_prior.json
```

Run Gen original TOF:

```bash
python scripts/run_gen_original_tof.py \
  --generated-pkl outputs/zero_target_runs/fr_tt/seed2024/codex_generation/generated_codex.pkl \
  --dataset fr --scenario tt --seed 2024 \
  --out-dir outputs/zero_target_runs/fr_tt/seed2024/gen_original_tof \
  --out-pkl outputs/zero_target_runs/fr_tt/seed2024/gen_original_tof/gen_tof.pkl \
  --cuda-visible-devices 0
```

Run the proposed main experiment (no ablation and no Causal-TOF):

```bash
python scripts/main_run_downstream_ad.py \
  --dataset fr --scenario tt --seed 2024 \
  --variant proposed_zero_target_causal_gss_codex \
  --input-root outputs/zero_target_runs \
  --out-root outputs/zero_target_runs \
  --device cuda --cuda-visible-devices 0
```

## Checks

```bash
pytest -q
python scripts/check_gen_main_data.py
csh doctor
```

`csh doctor --json` produces a machine-readable project report. New experiment
outputs belong in `outputs/`; do not overwrite historical formal artifacts in
`data/main_experiment/`.

## Project structure

| path | role |
| --- | --- |
| `causal_smart_home/causal_relation_prior_source.py` | Resolves a source-only causal prior. |
| `causal_smart_home/causal_gss_reweight.py` | Fuses source transitions and causal strengths. |
| `scripts/main_prepare_generation.py` | Builds a zero-target causal-GSS generation package. |
| `scripts/validate_and_pack_codex_generation.py` | Validates Codex JSONL and writes Gen pickle data. |
| `scripts/run_gen_original_tof.py` | Runs SmartGen's original two-stage filter. |
| `scripts/main_run_downstream_ad.py` | Runs one proposed downstream AD experiment. |
| `scripts/summarize_main_experiment.py` | Writes proposed per-seed summaries. |
| `scripts/check_project.py` | Checks layout, assets, and result metadata. |
| `data/main_experiment/` | Historical target-reference-assisted artifacts. |
| `outputs/zero_target_runs/` | New zero-target-data runs. |

## Rollback

The complete pre-redesign state is preserved outside this repository at:

```text
/home/heyang/projects/CausalSmartHome_checkpoints/20260711_before_target_guard_redesign/
```

Git tag:

```text
archive-before-target-guard-redesign-20260711
```
