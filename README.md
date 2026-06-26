# CausalSmartHome

## Overview

CausalSmartHome is a fused Gen + GCAD project for smart-home behavior
generation and anomaly detection. The main experiment matrix follows Gen's
anomaly-detection setup over FR/SP/US and three target contexts:
spring, night, and multiple. The central contribution is
causal-relation-enhanced GSS for GPT-5.5 behavior generation.

Causal-TOF is a post-TOF causal consistency enhancement component in the full
pipeline. It is evaluated as part of the proposed method and through one
ablation that removes this component.

## Proposed Method

The proposed method is:

`proposed_causal_gss_gpt55_causal_tof`

It combines causal relation prior construction, target-distribution constraint,
causal-reweighted GSS, GPT-5.5 generation, Gen original two-stage TOF,
Causal-TOF, and Gen built-in downstream AD.

The retained ablation is:

`ablation_no_causal_tof`

It uses the same causal-relation-enhanced GSS and GPT-5.5 generation path, then
removes the Causal-TOF component before downstream AD.

## Pipeline

```text
causal relation prior
-> target-distribution constraint
-> causal-reweighted GSS
-> GPT-5.5 generation
-> Gen original two-stage TOF
-> Causal-TOF
-> Gen built-in downstream AD
-> summary
```

## Main Results

The currently locked completed result is the SP-ST / SP-spring three-seed cell.
The latest GPU rerun after the Causal-TOF guard fix is stored under
`outputs/main_experiment_gpu_fix/sp_st/summary_standard/`.

| method | precision mean | recall mean | f1 mean | accuracy mean | fpr mean | fnr mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| proposed_causal_gss_gpt55_causal_tof | 0.953861 | 0.997711 | 0.975220 | 0.974588 | 0.048535 | 0.002289 |
| ablation_no_causal_tof | 0.840026 | 0.992216 | 0.898191 | 0.867903 | 0.256410 | 0.007784 |

The proposed method centers on causal-relation-enhanced GSS for GPT-5.5 behavior
generation. On SP-ST / SP-spring, the full pipeline with Causal-TOF achieves the best
three-seed mean performance. Removing the Causal-TOF component decreases mean
F1 from 0.975220 to 0.898191 and increases mean FPR from 0.048535 to 0.256410.

Per-seed SP-ST / SP-spring GPU rerun:

| seed | ablation F1 | proposed F1 | proposed device |
| --- | ---: | ---: | --- |
| 2024 | 0.741344 | 0.965517 | cuda |
| 2025 | 0.974565 | 0.981132 | cuda |
| 2026 | 0.978665 | 0.979012 | cuda |

The Gen paper/project anomaly-detection reference scores are vendored under
`outputs/reference_gen/anomaly_detection_pipeline_results/` and cover the 9
main cells:

| dataset | target context | Gen F1 |
| --- | --- | ---: |
| fr | spring | 0.861386 |
| fr | night | 0.969944 |
| fr | multiple | 0.932642 |
| sp | spring | 0.919057 |
| sp | night | 0.962482 |
| sp | multiple | 0.793970 |
| us | spring | 0.930290 |
| us | night | 0.876999 |
| us | multiple | 0.840492 |

## Reproduction

Use the frozen three-seed archive when reproducing the current main result:

```bash
bash outputs/main_experiment_frozen/sp_st_gpt55_proposed_3seed_20260623/run_reproduce_from_frozen.sh
```

The script reuses the frozen GPT-5.5 generated pkl files. It reruns Gen original
two-stage TOF, the w/o Causal-TOF downstream AD run, Causal-TOF, the proposed
method downstream AD run, and summary.

## Reproducibility Guardrails

GPU execution is required for Gen original TOF and Gen downstream AD. Do not
silently fall back to CPU when reproducing or extending the main experiments.
If a managed sandbox hides CUDA devices, rerun the experiment command with
elevated GPU access instead of changing the code path to CPU.

Causal relation weights must be normalized before sparse thresholding. The
GCAD-style gradient weights can be very small in raw scale; thresholding before
normalization can zero out all raw causal edges and reduce the pipeline to a
Gen transition-only GSS.

`build_causal_gss_prompt.py` defaults to adding causal edges and using
`guard-mode=downweight`. Causal-TOF keeps downweighted edges in the score audit,
but does not count them in the causal-violation penalty by default. This avoids
penalizing edges whose endpoints were already marked as overused by the target
distribution guard. Use `--penalize-downweighted-edges` only for diagnostic
experiments.

## Quick Checks

Run the unit tests and locked-result integrity check from the project root:

```bash
pytest -q
python -m causal_smart_home.cli check-gen-data
python -m causal_smart_home.cli check-recovery
```

The package also exposes the same utilities through the `csh` console entry
point after installation:

```bash
csh check-gen-data
csh check-recovery
csh summarize
```

Scenario aliases used by the scripts are `st` = spring, `tt` = night, and
`nt` = multiple. The full Gen main data requirement can be checked with
`python scripts/check_gen_main_data.py`.

## Project Structure

```text
causal_smart_home/
  causal_relation_adapter.py
  causal_relation_prior_source.py
  causal_gss.py
  causal_gss_reweight.py
  causal_tof.py
  gen_core/
  resources/gen_data/
scripts/
  build_causal_gss_prompt.py
  validate_and_pack_gpt55_generation.py
  run_gen_original_tof.py
  run_causal_tof.py
  run_gen_downstream_ad.py
  summarize_main_experiment.py
  freeze_main_experiment.py
outputs/main_experiment/
outputs/main_experiment_frozen/
outputs/reference_gen/
tests/
```
