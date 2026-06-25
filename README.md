# CausalSmartHome

## Overview

CausalSmartHome studies smart-home behavior generation and anomaly detection
over the canonical main experiment matrix:

```text
FR/SP/US x ST/TT/NT = 9 dataset-scenario cells
seeds = 2024, 2025, 2026
```

The main comparison is:

```text
original_gen_reference / SmartGen paper Table 3 SmartGen column
vs
proposed_causal_gss_gpt55_causal_tof
```

`ablation_no_causal_tof` is retained only for the Causal-TOF ablation table. It
is not the original Gen baseline and must not be used as the main baseline.

## Variants

`original_gen_reference` is the paper-reported SmartGen/Gen reference baseline
from SmartGen Table 3, SmartGen column. The reference values live in
`causal_smart_home/resources/reference/smartgen_table3_ad.json`. They report
precision, recall, and F1; they are not rerun outputs from this repository.

`proposed_causal_gss_gpt55_causal_tof` is:

```text
causal prior
-> target distribution guard
-> causal-reweighted GSS
-> GPT-5.5 generation package / validated generation
-> Gen original two-stage TOF
-> Causal-TOF
-> Gen downstream AD
```

`ablation_no_causal_tof` uses the same causal-reweighted GSS, GPT-5.5
generation, and Gen original two-stage TOF path, then skips Causal-TOF before
Gen downstream AD. It appears only in `ablation_causal_tof.*`.

## Current Status

The currently completed subset is SP-ST for seeds 2024, 2025, and 2026. That
historical SP-ST result is preserved, but it no longer represents the complete
main experiment. The other eight dataset-scenario cells are marked `MISSING`
until their GPT-5.5 generation, TOF, Causal-TOF, and downstream AD outputs are
actually produced.

Use the matrix status report to inspect what exists:

```bash
PYTHONPATH=. python scripts/run_main_experiment_matrix.py --dry-run --matrix all
```

## Main Commands

```bash
PYTHONPATH=. python scripts/run_main_experiment_matrix.py --dry-run --matrix all
PYTHONPATH=. python scripts/run_main_experiment_matrix.py --stage build_generation_package --matrix all
PYTHONPATH=. python scripts/run_main_experiment_matrix.py --stage validate_generation --matrix all
PYTHONPATH=. python scripts/run_main_experiment_matrix.py --stage downstream --matrix all
PYTHONPATH=. python scripts/summarize_main_experiment.py --matrix all
PYTHONPATH=. python scripts/summarize_main_experiment.py --matrix all --ablation
PYTHONPATH=. python scripts/freeze_main_experiment.py --matrix all
PYTHONPATH=. pytest -q
```

## Summary Outputs

The corrected summaries are written under `outputs/main_experiment/summary/`:

- `main_comparison_per_seed.*`
- `main_comparison_vs_gen.*`
- `main_comparison_aggregate.*`
- `ablation_causal_tof.*`
- `matrix_status_report.*`

Every main summary states that the baseline is SmartGen/Gen Table 3 reference,
not `ablation_no_causal_tof`. Missing proposed runs are shown as `MISSING`.

## Project Structure

```text
causal_smart_home/
  experiment_matrix.py
  causal_relation_adapter.py
  causal_relation_prior_source.py
  causal_gss.py
  causal_gss_reweight.py
  causal_tof.py
  gen_core/
  resources/gen_data/
  resources/reference/smartgen_table3_ad.json
scripts/
  run_main_experiment_matrix.py
  build_causal_gss_prompt.py
  build_gpt55_generation_package.py
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
