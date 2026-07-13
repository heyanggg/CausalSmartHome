# CausalSmartHome

## Paper experiment protocol (2026-07-14)

The maintained paper pipeline is target-aware and keeps SmartGen's Gen TOF and
downstream anomaly-detection protocols unchanged:

```text
SmartGen
-> gradient causal relation discovery
-> target-aware causal-prior adaptation
-> causal GSS
-> Codex generation
-> Gen original TOF (unchanged)
-> Causal-TOF consistency refinement
-> Gen downstream AD (unchanged)
```

The causal implementation is organized under:

```text
causal_smart_home/causal/
  discovery/gradient_gc.py
  adaptation/target_guard.py
  generation/causal_gss.py
  refinement/causal_tof.py
  evaluation/causal_metrics.py
  evaluation/generation_quality.py
```

Historical imports such as `causal_smart_home.causal_prior`,
`causal_smart_home.target_distribution_guard`, and
`causal_smart_home.causal_tof` remain supported.

Target normal behavior supplies an event-level device distribution. Each source
causal edge is adapted before GSS fusion:

```text
C_target(i,j) = C_source(i,j) * P_target(i) * P_target(j)
```

Each seed writes `target_adapted_causal_prior.json`, including before/after edge
statistics. Causal-TOF reports `causal_consistency_score`, the mean normalized
causal strength over ordered event pairs. Its higher-is-better utility is:

```text
final_score = alpha * reconstruction_loss
              - beta * causal_inconsistency
              - gamma * distribution_penalty
```

Gen's original TOF logic is not modified. Its current wrapper does not expose
per-sequence reconstruction losses, so that term is zero unless a separate
audited loss vector is explicitly supplied to Causal-TOF.

The four canonical ablation variants are:

| variant | causal GSS | Causal-TOF |
| --- | --- | --- |
| `baseline_gen` | no | no |
| `causal_gss_only` | yes | no |
| `causal_tof_only` | no | yes |
| `full_causal` | yes | yes |

All variants support FR/SP/US × ST/TT/NT. Results are stored per seed and are
never averaged by the summary scripts. Generation quality includes
`KL(target || synthetic)`, transition-matrix cosine similarity, and causal-graph
cosine similarity. Each variant also saves target/synthetic transition and
causal matrices in `case_study_matrices.json`.

Run checks first, then exactly one cell:

```bash
PY=/home/heyang/miniconda3/envs/smartguard_env/bin/python
$PY -m pytest -q

# Build target-aware causal GSS artifacts for SP-ST seed2024. Existing accepted
# Codex/Gen outputs may then be reused without overwriting historical results.
$PY scripts/main_prepare_generation.py \
  --dataset sp --scenario st --seed 2024 \
  --prior-json data/main_experiment/sp_st/seed2024/causal_gss/resolved_causal_relation_prior.json \
  --out-root outputs/main_runs

# Run only SP-ST seed2024, all four variants.
CUDA_VISIBLE_DEVICES=0 $PY scripts/main_run_ablation.py \
  --dataset sp --scenario st --seed 2024 \
  --input-root data/main_experiment \
  --out-root outputs/main_runs \
  --guarded-hints-json outputs/main_runs/sp_st/seed2024/causal_gss/causal_reweighted_gss_hints.json \
  --device cuda --cuda-visible-devices 0
```

Only after this cell's AD and generation-quality metrics are checked should the
same command be expanded to the remaining eight cells. New artifacts go under
`outputs/main_runs/`; existing `data/main_experiment/` results are read-only.

### SP-ST seed2024 validation result

The staged validation was executed on GPU 0 (RTX 3090) with the unchanged Gen
downstream AD protocol and 15 epochs. No other cell was run.

| variant | Precision | Recall | F1 | FPR |
| --- | ---: | ---: | ---: | ---: |
| `baseline_gen` | 0.858491 | 1.000000 | 0.923858 | 0.164835 |
| `causal_gss_only` | 0.588997 | 1.000000 | 0.741344 | 0.697802 |
| `causal_tof_only` | 0.824462 | 1.000000 | 0.903786 | 0.212912 |
| `full_causal` | 0.919192 | 1.000000 | 0.957895 | 0.087912 |

| variant | device KL ↓ | transition similarity ↑ | causal similarity ↑ |
| --- | ---: | ---: | ---: |
| `baseline_gen` | 1.183431 | 0.127651 | 0.133739 |
| `causal_gss_only` | 0.639495 | 0.748733 | 0.541435 |
| `causal_tof_only` | 1.175672 | 0.108294 | 0.135861 |
| `full_causal` | 0.566248 | 0.665461 | 0.453825 |

Mean causal consistency was `0.055326` for `causal_tof_only` and `0.111610`
for `full_causal`. The full method improves AD F1 and FPR over Gen while also
having the lowest device-distribution KL. The component-only results show that
the gain is not a trivial consequence of either causal component in isolation.

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

## Historical zero-target protocol

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
