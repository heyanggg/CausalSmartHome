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

## Current SP-ST Results

The completed SP-ST / SP-spring GPU rerun is stored locally under:

```text
outputs/main_experiment/sp_st/
outputs/main_experiment/summary/
```

Main results must be read seed by seed. Do not replace this table with an
average table, and do not report deltas against Gen.

| dataset | scenario | seed | Gen paper AD F1 | ablation_no_causal_tof F1 | proposed_causal_gss_codex_causal_tof F1 | proposed precision | proposed recall | proposed FPR | device |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sp | spring | 2024 | 0.919057 | 0.741344 | 0.965517 | 0.933333 | 1.000000 | 0.071429 | cuda |
| sp | spring | 2025 | 0.919057 | 0.974565 | 0.981132 | 0.962963 | 1.000000 | 0.038462 | cuda |
| sp | spring | 2026 | 0.919057 | 0.978665 | 0.979012 | 0.965287 | 0.993132 | 0.035714 | cuda |

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
