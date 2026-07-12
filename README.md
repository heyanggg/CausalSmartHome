# CausalSmartHome

## Required runtime

The repository root and the only supported Python interpreter for formal runs are:

```text
/home/heyang/projects/CausalSmartHome
/home/heyang/miniconda3/envs/smartguard_env/bin/python
```

Start every session and verify GPU 0 before running an experiment:

```bash
cd /home/heyang/projects/CausalSmartHome
nvidia-smi
/home/heyang/miniconda3/envs/smartguard_env/bin/python -c \
  'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

`torch.cuda.is_available()` must print `True`. Formal Gen TOF and downstream AD
runs must use `CUDA_VISIBLE_DEVICES=0`, and their result metadata must record
`device = cuda` and `requested_device = cuda`. CPU results are never formal
results. Do not use system Python, `/home/anaconda3/bin/python`, or another
Conda environment.

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

Run all commands from the repository root with the fixed interpreter above.

Prepare a source-only causal-GSS prompt package:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/heyang/miniconda3/envs/smartguard_env/bin/python \
scripts/main_prepare_generation.py \
  --dataset fr --scenario tt --seed 2024 \
  --out-root outputs/zero_target_runs
```

Validate and package Codex-authored JSONL:

```bash
/home/heyang/miniconda3/envs/smartguard_env/bin/python \
scripts/validate_and_pack_codex_generation.py \
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
CUDA_VISIBLE_DEVICES=0 \
/home/heyang/miniconda3/envs/smartguard_env/bin/python \
scripts/run_gen_original_tof.py \
  --generated-pkl outputs/zero_target_runs/fr_tt/seed2024/codex_generation/generated_codex.pkl \
  --dataset fr --scenario tt --seed 2024 \
  --out-dir outputs/zero_target_runs/fr_tt/seed2024/gen_original_tof \
  --out-pkl outputs/zero_target_runs/fr_tt/seed2024/gen_original_tof/gen_tof.pkl \
  --cuda-visible-devices 0
```

Run the proposed main experiment (no ablation and no Causal-TOF):

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/heyang/miniconda3/envs/smartguard_env/bin/python \
scripts/main_run_downstream_ad.py \
  --dataset fr --scenario tt --seed 2024 \
  --variant proposed_zero_target_causal_gss_codex \
  --input-root outputs/zero_target_runs \
  --out-root outputs/zero_target_runs \
  --device cuda --cuda-visible-devices 0
```

## Checks

```bash
/home/heyang/miniconda3/envs/smartguard_env/bin/python -m pytest -q
/home/heyang/miniconda3/envs/smartguard_env/bin/python scripts/check_gen_main_data.py
/home/heyang/miniconda3/envs/smartguard_env/bin/python scripts/check_project.py
/home/heyang/miniconda3/envs/smartguard_env/bin/python -m compileall -q causal_smart_home scripts tests
git diff --check
```

`csh doctor --json` produces a machine-readable project report. New experiment
outputs belong in `outputs/`; do not overwrite historical formal artifacts in
`data/main_experiment/`.

## Formal output and result status

All new formal artifacts belong under `outputs/zero_target_runs/`. Generation,
prompt construction, causal-GSS, and Gen TOF must not read target `trn.pkl`,
`vld.pkl`, `rs_vld.pkl`, `test.pkl`, `split_test.pkl`, downstream target normal,
or attack data. Only final downstream AD evaluation may load target normal and
attack data. Formal runs do not include Target Distribution Guard, Causal-TOF,
ablation, a generation-stage target pickle, or a target empirical distribution.

| group | seed | Precision | Recall | F1 | FPR | status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `fr_st` | 2024 | 0.778761 | 1.000000 | 0.875622 | 0.284091 | exceeds Gen reference 0.861386; selected best |
| `fr_tt` | 2024 | 0.977413 | 1.000000 | 0.988577 | 0.023109 | exceeds Gen reference 0.969944 |
| `fr_tt` | 2025 | 0.983471 | 1.000000 | 0.991667 | 0.016807 | exceeds Gen reference 0.969944; selected best |
| `fr_tt` | 2026 | 0.979424 | 1.000000 | 0.989605 | 0.021008 | exceeds Gen reference 0.969944 |
| `fr_nt` | 2024 | 0.000000 | 0.000000 | 0.000000 | 0.966667 | below Gen reference 0.932642 after three retained attempts; deferred |
| `sp_st` | 2024 | 0.825397 | 1.000000 | 0.904348 | 0.211538 | best retained; below Gen reference 0.919057 |
| `sp_st` | 2025 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | training-seed failure; deferred |
| `sp_tt` | 2024 | 0.979457 | 0.782532 | 0.869990 | 0.016413 | below Gen reference 0.962482 |
| `sp_tt` | 2025 | 0.865112 | 1.000000 | 0.927678 | 0.155920 | best retained; below Gen reference 0.962482; deferred |
| `sp_nt` | 2024 | 0.000000 | 0.000000 | 0.000000 | 0.753165 | two retained attempts; below Gen reference 0.793970 |
| `sp_nt` | 2025 | 0.000000 | 0.000000 | 0.000000 | 0.734177 | two retained attempts; below Gen reference 0.793970 |
| `sp_nt` | 2026 | 0.000000 | 0.000000 | 0.000000 | 0.753165 | two retained attempts; deferred |
| `us_st` | 2024 | 0.497705 | 0.970822 | 0.658051 | 0.979775 | best retained after two attempts; below Gen reference 0.930290 |
| `us_st` | 2025 | 0.497705 | 0.970822 | 0.658051 | 0.979775 | best retained after three source-only retries; below Gen reference 0.930290 |
| `us_st` | 2026 | 0.530519 | 1.000000 | 0.693254 | 0.884947 | best/current retained; below Gen reference 0.930290 |
| `us_tt` | 2024 | 0.379731 | 0.607809 | 0.467432 | 0.992822 | first retained attempt; below Gen reference 0.876999 |
| `us_tt` | 2025 | 0.403735 | 0.459374 | 0.429761 | 0.678438 | first retained additive-GSS attempt; below Gen reference 0.876999 |
| `us_nt` | 2024 | 0.000000 | 0.000000 | 0.000000 | 0.648177 | first retained attempt; below Gen reference 0.840492 |

Every run must report Precision, Recall, F1, FPR, and TP/TN/FP/FN. The current
group-level completion policy requires at least one formal seed to clearly
exceed its Gen reference F1. Other seeds should still be attempted where
practical, but repeated honest failures do not block the next group. Failed and
exploratory results must be retained with their parameters; results must never
be edited, fabricated, or selectively deleted.

`README_SELF.md` is a permanent, read-only historical archive. Documentation
synchronization, formatting, and bulk replacement must never modify it.

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
