# Internal Maintenance Notes

This file is for project maintainers and automation agents. The public-facing project summary is in `README.md`.

## Current Main Experiment

The current successful main experiment is:

```text
SP-ST GPT-5.5 generation, seeds 2024/2025/2026
```

Main output root:

```text
outputs/mainline_gpt55_generation/
```

Frozen archive:

```text
outputs/main_experiment_frozen/sp_st_gpt55_3seed_20260623
```

Do not delete the frozen archive. It is the long-term reproducibility copy of the current three-seed main experiment.

## Do Not Regenerate These Three Seeds

The GPT-5.5 JSONL content for seeds 2024, 2025, and 2026 is already generated, validated, packed, evaluated, summarized, and frozen.

For these seeds, use the archived files directly:

```text
generated_gpt55_clean.jsonl
generated_gpt55_clean.pkl
generation_report.json
validation_report.json
```

The generation metadata uses:

```json
{
  "generator": "gpt55_generation",
  "generation_model": "GPT-5.5",
  "api_llm": false,
  "manual_generation": true
}
```

`api_llm=false` is correct because the current three-seed files were not produced by an automated batch API call. Python was used only for validation, packing, and downstream pipeline execution.

## Reproduce From Frozen

To rerun the post-generation workflow from frozen pkl files:

```bash
bash outputs/main_experiment_frozen/sp_st_gpt55_3seed_20260623/run_reproduce_from_frozen.sh
```

The reproduce script reruns:

```text
GPT-5.5 generated pkl
  -> SmartGen original two-stage TOF
  -> raw_no_smartgen_tof AD
  -> mainline_smartgen_original_tof AD
  -> optional Causal-TOF
  -> mainline_smartgen_original_tof_plus_causal_tof AD
  -> summary
```

It writes to `reproduced_runs/` inside the frozen directory and does not overwrite the archived original results.

Downstream AD retrains neural models, so small metric differences can appear across GPU, driver, PyTorch, and CUDA environments.

## Three-Seed Aggregate

| variant | precision mean | recall mean | f1 mean | accuracy mean | fpr mean | fnr mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_no_smartgen_tof | 0.818177 | 1.000000 | 0.891371 | 0.862408 | 0.275183 | 0.000000 |
| mainline_smartgen_original_tof | 0.840026 | 0.992216 | 0.898191 | 0.867903 | 0.256410 | 0.007784 |
| mainline_smartgen_original_tof_plus_causal_tof | 0.953861 | 0.997711 | 0.975220 | 0.974588 | 0.048535 | 0.002289 |

## Causal-TOF vs SmartGen Original TOF

| seed | precision delta | recall delta | f1 delta | accuracy delta | fpr delta | fnr delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024 | +0.344337 | +0.000000 | +0.224173 | +0.313187 | -0.626374 | +0.000000 |
| 2025 | +0.012571 | +0.000000 | +0.006567 | +0.006868 | -0.013736 | +0.000000 |
| 2026 | -0.015403 | +0.016484 | +0.000347 | +0.000000 | +0.016484 | -0.016484 |

Across the three seeds, the post-TOF Causal-TOF variant has the best aggregate F1 and accuracy and the lowest aggregate FPR.

## Project Structure Notes

Current main-experiment files are concentrated in:

```text
causal_smart_home/
scripts/
tests/
outputs/mainline_gpt55_generation/
outputs/main_experiment_frozen/
outputs/gpt55_generation_package/
outputs/gcad_gss_stage4/*prompt/prior/guard/hints/package*
external_sources/
```

Keep `external_sources/`; it points to upstream projects used by wrappers and reproduction commands.

If expanding to a new scenario, keep the current mainline shape:

```text
GPT-5.5 generation
  -> SmartGen original two-stage TOF
  -> optional Causal-TOF
  -> SmartGen built-in downstream AD
```

The project no longer uses the old local sampling generator as a main-experiment path. Do not compare new GPT-5.5 main results against that removed generator path.
