# CausalSmartHome

## Overview

CausalSmartHome studies smart-home behavior generation and anomaly detection for
the SP-ST setting. The central contribution is causal-relation-enhanced GSS for
GPT-5.5 behavior generation.

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

| method | precision mean | recall mean | f1 mean | accuracy mean | fpr mean | fnr mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| proposed_causal_gss_gpt55_causal_tof | 0.953861 | 0.997711 | 0.975220 | 0.974588 | 0.048535 | 0.002289 |
| ablation_no_causal_tof | 0.840026 | 0.992216 | 0.898191 | 0.867903 | 0.256410 | 0.007784 |

The proposed method centers on causal-relation-enhanced GSS for GPT-5.5 behavior
generation. On SP-ST, the full pipeline with Causal-TOF achieves the best
three-seed mean performance. Removing the Causal-TOF component decreases mean
F1 from 0.975220 to 0.898191 and increases mean FPR from 0.048535 to 0.256410.

## Reproduction

Use the frozen three-seed archive when reproducing the current main result:

```bash
bash outputs/main_experiment_frozen/sp_st_gpt55_proposed_3seed_20260623/run_reproduce_from_frozen.sh
```

The script reuses the frozen GPT-5.5 generated pkl files. It reruns Gen original
two-stage TOF, the w/o Causal-TOF downstream AD run, Causal-TOF, the proposed
method downstream AD run, and summary.

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
tests/
```
