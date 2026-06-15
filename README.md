# Causal Smart Home

A non-invasive glue-layer prototype for stitching **SmartGuard**, **SmartGen** and **GCAD-style Granger causality**.

The project deliberately does **not** edit the original project code. The three original code snapshots are placed under `external_sources/` for reference/wrapping only:

- `external_sources/SmartGuard`
- `external_sources/SmartGen`
- `external_sources/GCAD`

No code from the failed experimental project is used or referenced in this project.

## Core idea

SmartGuard is strong at time-aware unsupervised smart-home anomaly detection, but it is trained on static normal routines. SmartGen can synthesize target-context behavior sequences for drift adaptation, but its graph-guided synthesis is mainly transition-frequency guidance. GCAD mines dynamic Granger causal patterns from multivariate time series. This glue layer converts discrete behavior sequences into event tensors, mines action/subsequence-level causal priors, injects those priors into SmartGen prompts, then post-filters generated sequences before SmartGuard retraining.

The result is a pipeline:

```text
normal behavior sequences
  -> SmartGen TSS/SSC/GSS candidates
  -> event tensor bridge
  -> GCAD-style causal prior
  -> causal prompt hints + causal post-filter
  -> SmartGuard retraining and evaluation
```

## Installation

```bash
cd CausalSmartHome
python -m pip install -e .[dev]
```

## Quick test

```bash
python -m pytest -q
python -m causal_smart_home.cli demo --out-dir outputs/demo --epochs 5 --lag 3
```

The demo uses toy smart-home sequences because the provided tarball contains Python files only, not the full FR/SP/US/SmartSense datasets.

## Using with real SmartGen/SmartGuard data

The glue layer expects SmartGuard/SmartGen numeric sequence pickles where each sequence is a flattened list of quadruples:

```text
[day, hour_slot, device_id, action_id, day, hour_slot, device_id, action_id, ...]
```

1. Build a causal prior from normal training sequences:

```bash
csh build-prior \
  --train-pkl path/to/split_trn.pkl \
  --out-dir outputs/fr_spring \
  --lag 4 \
  --epochs 80 \
  --level action \
  --sparse-threshold 0.001
```

2. Build a causal-aware SmartGen prompt:

```bash
csh prompt \
  --prior-json outputs/fr_spring/causal_prior.json \
  --compressed-pkl path/to/trn_day_0_SPPC_th=0.918.pkl \
  --device-info-json path/to/device_info.json \
  --original-context winter \
  --new-context spring \
  --out-prompt outputs/fr_spring/causal_prompt_day0.txt
```

3. After SmartGen/LLM produces sequences, filter them causally:

```bash
csh filter \
  --prior-json outputs/fr_spring/causal_prior.json \
  --generated-pkl path/to/generated_seq.pkl \
  --out-pkl outputs/fr_spring/generated_causal_kept.pkl \
  --out-scores outputs/fr_spring/causal_scores.json \
  --min-coverage 0.5
```

4. Merge `generated_causal_kept.pkl` with the target-context train split and run the unmodified SmartGuard training script in `external_sources/SmartGuard` or in your cloned SmartGuard repository.

## Modules

- `schema.py`: typed behavior and behavior-sequence representation.
- `event_tensor.py`: converts discrete behavior events into multivariate time-series tensors.
- `causal_prior.py`: compact GCAD-style gradient causal miner and serializable causal prior.
- `gcad_adapter.py`: wrapper/fallback for GCAD-style mining.
- `smartgen_adapter.py`: wrapper utilities for SmartGen data conventions.
- `smartguard_adapter.py`: subprocess wrapper for unmodified SmartGuard scripts.
- `causal_prompt.py`: SmartGen prompt builder with causal JSON hints.
- `causal_filter.py`: causal consistency scoring and filtering.
- `pipeline.py`: end-to-end orchestration.

## Real experiment plan

Main datasets: FR, SP and US, following SmartGen context shifts:

- seasonal transition: winter -> spring
- time-schedule transition: daytime -> night
- occupancy transition: single -> multiple

Main anomaly tests: SmartGuard's SD, MD, DM and DD injected attack categories.

Compare:

1. SmartGuard trained only on original normal data.
2. SmartGuard retrained with SmartGen data.
3. SmartGuard retrained with SmartGen data after SmartGen TOF.
4. Proposed glue: SmartGen + causal prompt hints + causal post-filter + SmartGuard.
5. Ablations without causal prompt, without causal filter, device-level vs action-level prior, varying lag.

## Notes

- This package is a research prototype and a glue implementation, not a replacement of the three original projects.
- The compact miner in `causal_prior.py` is intentionally small so the glue can be unit-tested. For full GCAD reproduction, prepare `train.csv`/`test.csv` folders and run the original GCAD CLI in `external_sources/GCAD`.
