# CausalSmartHome

CausalSmartHome evaluates whether causal structure can improve SmartGen-style target-context smart-home behavior generation and downstream anomaly detection.

The current main result is the SP-ST GPT-5.5 generation three-seed experiment for seeds 2024, 2025, and 2026.

## Main Pipeline

```text
GCAD causal prior
  -> target-distribution downweight guard
  -> multiplicative causal-reweighted GSS
  -> guarded causal-reweighted GSS prompt
  -> GPT-5.5 generation
  -> SmartGen original two-stage TOF
  -> optional Causal-TOF
  -> SmartGen built-in downstream AD
  -> summary
```

## Method Components

- GCAD causal prior: mines causal structure from source-context behavior data.
- Guarded causal-reweighted GSS: combines transition evidence, causal hints, and target-distribution guards before prompting generation.
- GPT-5.5 generation: produces target-context normal behavior sequences for downstream SmartGen training augmentation.
- SmartGen original two-stage TOF: restores SmartGen's original evaluation pipeline so the downstream comparison is aligned with the upstream method.
- Optional Causal-TOF: applies a post-TOF causal weighting/resampling step as an ablation.
- SmartGen built-in downstream AD: evaluates synthetic target-context data with SmartGen's anomaly detection pipeline.

SmartGen original two-stage TOF is included for evaluation alignment. It is not the main contribution being evaluated. The research focus is GCAD-guided causal GSS, GPT-5.5 target-context behavior generation, and optional post-TOF Causal-TOF.

## SP-ST GPT-5.5 Three-Seed Result

Aggregate over seeds 2024, 2025, and 2026:

| variant | precision mean | recall mean | f1 mean | accuracy mean | fpr mean | fnr mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_no_smartgen_tof | 0.818177 | 1.000000 | 0.891371 | 0.862408 | 0.275183 | 0.000000 |
| mainline_smartgen_original_tof | 0.840026 | 0.992216 | 0.898191 | 0.867903 | 0.256410 | 0.007784 |
| mainline_smartgen_original_tof_plus_causal_tof | 0.953861 | 0.997711 | 0.975220 | 0.974588 | 0.048535 | 0.002289 |

Per-seed, aggregate, and seed-delta summaries are under:

```text
outputs/mainline_gpt55_generation/gen_builtin_ad/
```

The frozen archive for the current main experiment is under:

```text
outputs/main_experiment_frozen/
```

## Repository Layout

```text
causal_smart_home/                  Core causal, TOF, schema, and SmartGen integration helpers
scripts/                            Mainline build, validation, TOF, AD, summary, and freeze scripts
tests/                              Unit tests for the current pipeline helpers
outputs/mainline_gpt55_generation/  Current SP-ST GPT-5.5 three-seed outputs
outputs/main_experiment_frozen/     Frozen reproducibility archive
outputs/gpt55_generation_package/   GPT-5.5 prompt/schema packages
external_sources/                   Local links to upstream SmartGen/GCAD checkouts
```

## Reproduce

Use the frozen archive's reproduce script to rerun TOF, Causal-TOF, downstream AD, and summary from the frozen generated pkl files. This does not regenerate GPT-5.5 JSONL content.

```bash
bash outputs/main_experiment_frozen/sp_st_gpt55_3seed_20260623/run_reproduce_from_frozen.sh
```

Downstream AD retrains neural models, so small metric differences can appear across GPU, driver, PyTorch, and CUDA environments.

## Test

```bash
cd /home/heyang/projects/CausalSmartHome
PYTHONPATH=. pytest -q tests
```

Expected: `18 passed`.
