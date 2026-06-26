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
main experiment.

Gen/SmartGen source data were migrated from `/home/heyang/projects/SmartGen`.
The migration accepts source aliases such as `night`/`nighttime`,
`multi`/`multiple`, `train.pkl`/`trn.pkl`, and `rs_vld.pkl`/`vld.pkl`, and
records the resolved alias in the manifest.

After the alias-aware migration, all 9 dataset-scenario cells have the required
SmartGen data/checkpoints for package construction and downstream evaluation.
The remaining incomplete cells are blocked by GPT-5.5 generation only
(`GENERATION_MISSING`).

For the NT (`single -> multiple`) cells, SmartGen's own anomaly-detection code
does not require `IoT_data/<dataset>/multiple/trn.pkl` or `vld.pkl`; the
`multiple` branch trains/thresholds on the generated/filter-true data. The main
package path therefore requires the source `single/trn.pkl`, target
`multiple/test.pkl`, target `multiple/split_test.pkl`, downstream attack/test
files, checkpoints, and dictionary. Generated/filter-data files are not copied
as normal input data.

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
PYTHONPATH=. python scripts/migrate_gen_experiment_data.py --source /home/heyang/projects/SmartGen --target /home/heyang/projects/CausalSmartHome --matrix all --dry-run --write-manifest
PYTHONPATH=. python scripts/summarize_main_experiment.py --matrix all
PYTHONPATH=. python scripts/summarize_main_experiment.py --matrix all --ablation
PYTHONPATH=. python scripts/freeze_main_experiment.py --matrix all
PYTHONPATH=. pytest -q
```

`build_generation_package` writes GPT-5.5 input packages for all 27
dataset-scenario-seed cells under:

```text
outputs/main_experiment/gpt55_generation_packages/<dataset>/<scenario>/seed<seed>/
```

The current package index is:

```text
outputs/main_experiment/gpt55_generation_packages/generation_package_index.json
outputs/main_experiment/gpt55_generation_packages/generation_package_index.md
outputs/main_experiment/gpt55_generation_packages/README_SEND_TO_GPT55.md
```

GPT-5.5 JSONL outputs are expected later at
`outputs/main_experiment/gpt55_generation/<dataset>_<scenario>/seed<seed>/generated_gpt55_clean.jsonl`.
The package files are prompts and schemas only; they are not generated results.

## Summary Outputs

The corrected summaries are written under `outputs/main_experiment/summary/`:

- `main_comparison_per_seed.*`
- `main_comparison_vs_gen.*`
- `main_comparison_aggregate.*`
- `ablation_causal_tof.*`
- `matrix_status_report.*`
- `outputs/data_migration/gen_data_migration_manifest.json`
- `outputs/data_migration/gen_data_migration_report.md`

Every main summary states that the baseline is SmartGen/Gen Table 3 reference,
not `ablation_no_causal_tof`. Missing proposed runs carry the matrix status,
such as `GENERATION_MISSING` or `MISSING_DATA`.

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
