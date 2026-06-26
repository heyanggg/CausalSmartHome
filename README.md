# CausalSmartHome

## Overview

CausalSmartHome is a Gen + GCAD fused project for smart-home behavior
generation and anomaly detection. The main experiment matrix follows Gen's
FR/SP/US anomaly-detection setup over three target contexts:

```text
FR/SP/US x spring/night/multiple
```

The project design is fixed as:

```text
causal relation prior
-> target-distribution guard
-> causal-reweighted GSS
-> Codex generation
-> Gen original two-stage TOF
-> Causal-TOF
-> Gen built-in downstream AD
-> per-seed summary
```

Causal-TOF is one step of the main experiment pipeline. It is not a separate
method. The retained ablation, `ablation_no_causal_tof`, removes this
pipeline step only to show what the full pipeline loses without it.

The proposed method name used by new runs is:

```text
proposed_causal_gss_codex_causal_tof
```

Historical proposed-method names are normalized to this Codex method name by
the summary script.

## Current Completed Results

The completed GPU runs are stored locally under:

```text
outputs/main_experiment/fr_st/
outputs/main_experiment/fr_tt/
outputs/main_experiment/sp_st/
outputs/main_experiment/sp_tt/
outputs/main_experiment/sp_nt/
outputs/main_experiment/summary/
```

Main results must be read seed by seed. Do not replace this table with an
average table, and do not report deltas against Gen.

| dataset | scenario | seed | Gen paper AD F1 | ablation_no_causal_tof F1 | proposed_causal_gss_codex_causal_tof F1 | proposed precision | proposed recall | proposed FPR | device |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fr | spring | 2024 | 0.861386 | 0.956522 | 0.977778 | 0.956522 | 1.000000 | 0.045455 | cuda |
| fr | spring | 2025 | 0.861386 | 0.814815 | 0.977778 | 0.956522 | 1.000000 | 0.045455 | cuda |
| fr | spring | 2026 | 0.861386 | 0.956522 | 0.983240 | 0.967033 | 1.000000 | 0.034091 | cuda |
| fr | night | 2024 | 0.969944 | 0.911005 | 0.993737 | 0.987552 | 1.000000 | 0.012605 | cuda |
| sp | spring | 2024 | 0.919057 | 0.741344 | 0.965517 | 0.933333 | 1.000000 | 0.071429 | cuda |
| sp | spring | 2025 | 0.919057 | 0.974565 | 0.981132 | 0.962963 | 1.000000 | 0.038462 | cuda |
| sp | spring | 2026 | 0.919057 | 0.978665 | 0.979012 | 0.965287 | 0.993132 | 0.035714 | cuda |
| sp | night | 2024 | 0.962482 | 0.786219 | 0.962482 | 0.927678 | 1.000000 | 0.077960 | cuda |
| sp | night | 2025 | 0.962482 | 0.962482 | 0.962190 | 0.927639 | 0.999414 | 0.077960 | cuda |
| sp | night | 2026 | 0.962482 | 0.841575 | 0.962190 | 0.927639 | 0.999414 | 0.077960 | cuda |
| sp | multiple | 2024 | 0.793970 | 0.940476 | 0.948949 | 0.902857 | 1.000000 | 0.107595 | cuda |
| sp | multiple | 2025 | 0.793970 | 0.948949 | 0.948949 | 0.902857 | 1.000000 | 0.107595 | cuda |
| sp | multiple | 2026 | 0.793970 | 0.943284 | 0.948949 | 0.902857 | 1.000000 | 0.107595 | cuda |

Gen paper/project anomaly-detection reference scores used for parallel
comparison:

| dataset | target context | Gen paper AD F1 |
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

The small Gen reference summaries are tracked under
`outputs/reference_gen/`. Large local pkl/checkpoint/experiment artifacts are
ignored by git.

## Running The Pipeline

Scenario aliases:

```text
st = spring
tt = night
nt = multiple
```

Core scripts:

```text
scripts/build_causal_gss_prompt.py
scripts/build_codex_generation_package.py
scripts/validate_and_pack_codex_generation.py
scripts/run_gen_original_tof.py
scripts/run_causal_tof.py
scripts/run_gen_downstream_ad.py
scripts/summarize_main_experiment.py
scripts/check_gen_main_data.py
```

GPU execution is required for Gen original TOF and Gen downstream AD. Results
must record `device = cuda` and `requested_device = cuda`. If a managed sandbox
hides CUDA devices, rerun the experiment command with GPU access; do not change
the experiment to CPU fallback.

## Guardrails

- Keep Causal-TOF in the proposed pipeline. It is part of the main method flow.
- List all seed results separately. Mean/std tables are not the main result.
- Compare against Gen paper AD scores by listing them side by side. Do not use
  delta tables as the comparison output.
- Normalize GCAD causal weights before sparse thresholding. Otherwise raw
  causal edges can be zeroed out and the GSS degenerates into a transition-only
  graph.
- `build_causal_gss_prompt.py` should add causal edges by default and use
  `guard-mode=downweight`.
- Causal-TOF keeps downweighted edges in the audit fields, but does not count
  `guard_action=downweight` edges in the causal-violation penalty by default.
  Use `--penalize-downweighted-edges` only for diagnostics.
- FR-spring target normal data contains Gen legacy `Other + None:location`
  rows. The Codex validator allows this original-format pairing; replacing it
  with dictionary-pure `Other:*` actions leaves many target normals uncovered
  and raises false positives.
- For FR-spring, Causal-TOF should use the default weight mode. Seed2025 needs
  a larger 200-row pre-TOF generation to stabilize the spring 80/20 validation
  split; final pre-TOF counts are 125 / 200 / 200 for seeds 2024 / 2025 / 2026,
  Gen TOF keeps 117 / 188 / 183 rows, and Causal-TOF keeps the same row counts.
- FR-night was run for seed2024 only by request. The key rule is temporal:
  target normal events occur in hour slots `0/1/6/7`, while the attack set moves
  otherwise similar behavior into `2/3/4/5`. Generate night normal rows only in
  `0/1/6/7`; seed2024 used 300 pre-TOF rows, Gen TOF kept 277, and Causal-TOF
  default weight mode kept 277.
- In SP-multiple, SmartGen's own gpt-4o synthetic source has 100 raw rows
  before TOF/filtering, but its downstream AD baseline uses the full filtered
  multiple-context synthetic set for both training and threshold calibration
  instead of the spring/night 80/20 generated split. Mirror that protocol.
- SP-multiple cannot simply copy SP-spring or SP-night generation. Match the
  multiple target-normal shape with 100 pre-TOF variable-length sequences
  spanning short 1-9 event behavior, include rare-but-normal target devices
  such as Other/SmartPlug/Projector/SmartLock/GarageDoor/Washer, and keep
  Television very scarce because the attack set is Television-only under the
  device-id AD model.
- For SP-multiple Causal-TOF, the stable setting is filter mode with
  `min_weight=0.05` and the default `penalize_downweighted_edges=false`. This
  avoids harmful duplicate resampling while still removing low causal-weight
  rows. The final SP-multiple Gen TOF counts are 97 / 100 / 100 for seeds
  2024 / 2025 / 2026, and Causal-TOF keeps 90 / 93 / 94 rows.

## Checks

Run from the project root:

```bash
pytest -q
python scripts/check_gen_main_data.py
csh summarize --runs-root outputs/main_experiment --out-dir outputs/main_experiment/summary
```

`scripts/check_gen_main_data.py` verifies the local FR/SP/US x
spring/night/multiple Gen data required by the main experiments.

## Project Structure

```text
causal_smart_home/
  causal_prior.py
  causal_relation_adapter.py
  causal_relation_prior_source.py
  causal_gss.py
  causal_gss_reweight.py
  causal_tof.py
  gen_core/
  resources/gen_data/
scripts/
  build_causal_gss_prompt.py
  build_codex_generation_package.py
  validate_and_pack_codex_generation.py
  run_gen_original_tof.py
  run_causal_tof.py
  run_gen_downstream_ad.py
  summarize_main_experiment.py
  check_gen_main_data.py
outputs/reference_gen/
tests/
```
