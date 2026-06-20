# Task 12: GCAD-GSS Prompt Enhancement Stage 3

## Scope

This document records the FR-ST GCAD-GSS prompt enhancement experiments.

Constraints:

- Do not modify SmartGen source code.
- Do not create a new project.
- Run FR winter -> spring only.
- Keep SmartGen source data, target context, TOF path, and downstream AD wrapper fixed.
- The controlled variable is the prompt: original SmartGen prompt vs GCAD-GSS enhanced prompt.
- Because the API environment is limited, use `codex-calibrated` as the fixed GPT-style sequence generation backend for future experiments.

## Prompt Wrapper

New modules:

- `causal_smart_home/causal_gss.py`
- `causal_smart_home/causal_prompt_adapter.py`

New prompt builder:

- `scripts/build_gcad_gss_prompt.py`

The wrapper learns a device-level GCAD prior from source real normal data, extracts top device-level causal edges, maps IDs to readable device names, formats soft causal hints, and inserts them after the original SmartGen GSS hints.

The original GSS hints are retained. The causal hints are soft constraints, not hard rules. The core wording is:

```text
The following device-level causal patterns are common in the user's historical behavior. When these devices appear together, preserve their usual temporal order unless the target context explicitly suggests a change.
```

Prompt-check artifacts:

```text
outputs/gcad_gss/fr_st_codex_calibrated_v3/fr_st_prompt_check/
```

The directory contains prompt diffs, top causal edges, and causal hint JSON files.

## Stage 3A Command

```bash
/home/heyang/miniconda3/envs/smartguard_env/bin/python scripts/run_stage3a_gcad_gss_fr_st.py \
  --stage all \
  --groups original,enhanced \
  --offline-generator codex-calibrated \
  --samples-per-category 6 \
  --no-reuse-existing-original \
  --output-tag fr_st_codex_calibrated_v3 \
  --sparse-threshold 0.001 \
  --epochs 80 \
  --sample-limit 64 \
  --seed 2024 \
  --require-cuda
```

Outputs:

```text
outputs/gcad_gss/fr_st_codex_calibrated_v3/fr_st_original/
outputs/gcad_gss/fr_st_codex_calibrated_v3/fr_st_enhanced/
outputs/gcad_gss/fr_st_codex_calibrated_v3/fr_st_stage3a_summary.csv
outputs/gcad_gss/fr_st_codex_calibrated_v3/fr_st_stage3a_summary.md
```

`codex-calibrated` is the fixed reproducible GPT-style text generator for this project under the current API constraints. It uses the existing SmartGen FR-ST TOF baseline as a style bank for sequence length, device/action diversity, and coarse transition variety. It then emits SmartGen-format textual responses and still goes through SmartGen Extract, Transnum, security_check, and TOF.

This mode should be reported as Codex-calibrated generation. It should not be described as a fully independent external LLM/API generation result.

## Stage 3A Results

| Group | Raw | TOF kept | Causal coverage | Violation | Low evidence | Action JS | Device JS | Transition JS | Avg len |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 222 | 207 | 0.5532 | 0.0217 | 0.4251 | 0.4784 | 0.2457 | 0.7891 | 4.9855 |
| enhanced | 222 | 209 | 0.5221 | 0.0760 | 0.4019 | 0.4234 | 0.2132 | 0.8027 | 5.3876 |

Interpretation:

- Better with enhanced prompt: low evidence rate, action JS to target, device JS to target, TOF kept rate.
- Worse with enhanced prompt: causal coverage, violation rate, transition JS to target.
- Stage 3A alone supports a cautious claim: GCAD-GSS enhanced prompt improves several generation-quality indicators, but not all causal/distribution metrics monotonically improve.

## Stage 3B Command

Stage 3B was run after Stage 3A as a downstream AD sanity check. It still uses FR-ST only and does not invoke SmartGuard.

```bash
/home/heyang/miniconda3/envs/smartguard_env/bin/python scripts/run_stage3b_ad_gcad_gss_fr_st.py \
  --stage3a-tag fr_st_codex_calibrated_v3 \
  --epochs 15 \
  --seed 2024 \
  --device cuda \
  --cuda-visible-devices 0
```

Outputs:

```text
outputs/gcad_gss/fr_st_stage3b_ad/fr_st_codex_calibrated_v3/metrics.csv
outputs/gcad_gss/fr_st_stage3b_ad/fr_st_codex_calibrated_v3/metrics.md
outputs/gcad_gss/fr_st_stage3b_ad/fr_st_codex_calibrated_v3/metrics.json
```

## Stage 3B Results

| Group | Precision | Recall | F1 | FPR | FNR | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original prompt | 0.6241 | 1.0000 | 0.7686 | 0.6023 | 0.0000 | 0.6989 |
| enhanced prompt | 0.7521 | 1.0000 | 0.8585 | 0.3295 | 0.0000 | 0.8352 |

Delta enhanced minus original:

- Precision: +0.1280
- Recall: +0.0000
- F1: +0.0900
- FPR: -0.2727
- Accuracy: +0.1364

Interpretation:

Under this controlled wrapper setting, GCAD-GSS enhanced prompt improves downstream SmartGen Transformer Autoencoder AD performance relative to original prompt. The main gain is lower false positive rate without losing recall.

## Cautious Claim

Recommended wording:

```text
On FR-ST, under a controlled Codex-calibrated GPT-style generation backend with unchanged SmartGen TOF and Transformer Autoencoder evaluation, the GCAD-GSS enhanced prompt improves several generation-quality indicators and yields a stronger downstream AD sanity-check result than the original prompt.
```

Avoid claiming:

- GCAD-GSS universally improves downstream AD.
- The result is already validated across all SmartGen settings.
- The `codex-calibrated` setting is equivalent to a fully independent external LLM/API run.

## Next Work

1. Run multi-seed repeats for `fr_st_codex_calibrated_v3`.
2. Reuse the saved original prompt arm when the generator protocol, seed, TOF, target test, and AD settings are unchanged.
3. Rerun the original prompt arm only when changing generator protocol, seed set, data split, or evaluation settings.
4. Compare the Codex-calibrated GCAD-GSS results against the SmartGen paper table as the reported original SmartGen baseline, without rerunning the paper baseline unless a reviewer explicitly requires local reproduction.
5. Extend to SP-ST and US-ST only after FR-ST repeat stability is confirmed.
