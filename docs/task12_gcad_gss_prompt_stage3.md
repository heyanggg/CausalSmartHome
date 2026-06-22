# Task 12: GCAD-GSS Prompt Enhancement Stage 3

## Scope

This document records the FR-ST and SP-ST GCAD-GSS prompt enhancement experiments.

Constraints:

- Do not modify SmartGen source code.
- Do not create a new project.
- Completed FR winter -> spring first; then extended the same Codex-calibrated protocol to SP daytime -> spring.
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

Single seed 2024:

| Group | Raw | TOF kept | Causal coverage | Violation | Low evidence | Action JS | Device JS | Transition JS | Avg len |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 222 | 207 | 0.5532 | 0.0217 | 0.4251 | 0.4784 | 0.2457 | 0.7891 | 4.9855 |
| enhanced | 222 | 209 | 0.5221 | 0.0760 | 0.4019 | 0.4234 | 0.2132 | 0.8027 | 5.3876 |

Interpretation:

- Better with enhanced prompt: low evidence rate, action JS to target, device JS to target, TOF kept rate.
- Worse with enhanced prompt: causal coverage, violation rate, transition JS to target.
- Stage 3A alone supports a cautious claim: GCAD-GSS enhanced prompt improves several generation-quality indicators, but not all causal/distribution metrics monotonically improve.

## Stage 3A Multi-Seed Results

Repeated with the same `codex-calibrated` generator protocol, `samples_per_category=6`, sparse threshold, TOF path, target real data, and quality settings for seeds 2024, 2025, and 2026.

Outputs:

```text
outputs/gcad_gss/fr_st_codex_calibrated_multiseed_stage3a_summary.csv
outputs/gcad_gss/fr_st_codex_calibrated_multiseed_stage3a_summary.md
```

Delta means are enhanced minus original:

| Metric | Mean delta | Std |
| --- | ---: | ---: |
| causal coverage | -0.0407 | 0.0083 |
| low evidence rate | -0.0139 | 0.0090 |
| violation rate | +0.0546 | 0.0041 |
| action JS to target | -0.0536 | 0.0029 |
| device JS to target | -0.0330 | 0.0006 |
| transition JS to target | +0.0167 | 0.0048 |
| TOF kept rate | -0.0045 | 0.0664 |

Per-seed deltas:

| Seed | Low evidence delta | Action JS delta | Device JS delta | F1 direction later |
| --- | ---: | ---: | ---: | --- |
| 2024 | -0.0232 | -0.0550 | -0.0325 | positive |
| 2025 | -0.0052 | -0.0555 | -0.0336 | positive |
| 2026 | -0.0133 | -0.0502 | -0.0329 | slightly negative |

Interpretation:

- Stable in 3/3 seeds: enhanced lowers low evidence rate, action JS to target, and device JS to target.
- Stable in the opposite direction in 3/3 seeds: enhanced lowers causal coverage and raises violation rate.
- Transition JS also worsens in 3/3 seeds.
- This supports a generation-quality claim focused on low-evidence and action/device distribution alignment, not a blanket causal-quality improvement claim.

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

Single seed 2024:

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

## Stage 3B Multi-Seed Results

Stage 3B was run for the same three Stage 3A tags:

```text
fr_st_codex_calibrated_v3
fr_st_codex_calibrated_seed2025
fr_st_codex_calibrated_seed2026
```

Summary outputs:

```text
outputs/gcad_gss/fr_st_codex_calibrated_multiseed_stage3b_summary.csv
outputs/gcad_gss/fr_st_codex_calibrated_multiseed_stage3b_summary.json
outputs/gcad_gss/fr_st_codex_calibrated_multiseed_stage3b_summary.md
```

Delta means are enhanced minus original:

| Metric | Mean delta | Std |
| --- | ---: | ---: |
| precision | +0.0520 | 0.0702 |
| recall | +0.0000 | 0.0000 |
| F1 | +0.0360 | 0.0500 |
| FPR | -0.1061 | 0.1551 |
| accuracy | +0.0530 | 0.0776 |

Per-seed AD deltas:

| Seed | F1 delta | FPR delta | Accuracy delta |
| --- | ---: | ---: | ---: |
| 2024 | +0.0900 | -0.2727 | +0.1364 |
| 2025 | +0.0268 | -0.0795 | +0.0398 |
| 2026 | -0.0087 | +0.0341 | -0.0170 |

Interpretation:

- Stage 3B mean delta remains positive for F1 and accuracy, and negative for FPR.
- The AD sanity-check improvement is not fully stable across seeds: seed 2026 is slightly worse for enhanced prompt.
- Treat Stage 3B as encouraging but seed-sensitive. Do not claim robust downstream AD improvement until more seeds or SP-ST/US-ST confirm the pattern.

## SP-ST Extension

SP-ST uses the same CausalSmartHome wrapper path and does not modify SmartGen source code. The actual SmartGen SP source/target paths are `sp/daytime -> sp/spring`.

Important SP-ST details:

- Generation backend: local `codex-calibrated`, not an external API LLM.
- SmartGen Extract, Transnum, security_check, TOF, and Transformer Autoencoder AD wrapper are unchanged.
- Seeds: 2024, 2025, 2026.
- Raw scale: 42 sequences per arm per seed, because SP daytime has 7 prompt categories and the fixed generator uses 6 samples per category.
- GCAD sparse threshold: `0.0`. A threshold of `0.001` removes all SP device-level edges because the largest learned SP edge is around `1.8e-4`.
- Stage 3B was run on CPU because the AD wrapper reported CUDA unavailable inside `run_smartgen_anomaly_experiment`; both arms and all seeds used the same device setting.

Stage 3A command pattern:

```bash
/home/heyang/miniconda3/envs/smartguard_env/bin/python scripts/run_stage3a_gcad_gss_sp_st.py \
  --stage all \
  --groups original,enhanced \
  --offline-generator codex-calibrated \
  --samples-per-category 6 \
  --seed 2024 \
  --output-tag sp_st_codex_calibrated_seed2024 \
  --epochs 80 \
  --sample-limit 64 \
  --force-tof
```

Stage 3B command pattern:

```bash
/home/heyang/miniconda3/envs/smartguard_env/bin/python scripts/run_stage3b_ad_gcad_gss_sp_st.py \
  --stage3a-tag sp_st_codex_calibrated_seed2024 \
  --seed 2024 \
  --epochs 15 \
  --device cpu \
  --cuda-visible-devices 0
```

Summary outputs:

```text
outputs/gcad_gss/sp_st_codex_calibrated_multiseed_stage3a_summary.csv
outputs/gcad_gss/sp_st_codex_calibrated_multiseed_stage3a_summary.json
outputs/gcad_gss/sp_st_codex_calibrated_multiseed_stage3a_summary.md
outputs/gcad_gss/sp_st_codex_calibrated_multiseed_stage3b_summary.csv
outputs/gcad_gss/sp_st_codex_calibrated_multiseed_stage3b_summary.json
outputs/gcad_gss/sp_st_codex_calibrated_multiseed_stage3b_summary.md
```

## SP-ST Stage 3A Multi-Seed Results

Delta means are enhanced minus original:

| Metric | Mean delta | Std |
| --- | ---: | ---: |
| low evidence rate | -0.0746 | 0.0295 |
| action JS to target | +0.0096 | 0.0089 |
| device JS to target | +0.0078 | 0.0052 |
| causal coverage | +0.0502 | 0.0229 |
| violation rate | -0.0012 | 0.0011 |
| transition JS to target | +0.0020 | 0.0035 |
| TOF kept rate | -0.0079 | 0.0364 |

Per-seed deltas:

| Seed | Low evidence | Action JS | Device JS | Coverage | Violation | Transition JS | TOF kept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024 | -0.1067 | +0.0195 | +0.0128 | +0.0578 | -0.0023 | +0.0059 | -0.0476 |
| 2025 | -0.0683 | +0.0072 | +0.0082 | +0.0683 | -0.0014 | +0.0009 | +0.0238 |
| 2026 | -0.0488 | +0.0022 | +0.0025 | +0.0244 | +0.0000 | -0.0009 | +0.0000 |

Interpretation:

- Stable improvements in 3/3 seeds: lower low evidence rate and higher causal coverage.
- Violation rate is slightly lower in 2/3 seeds and unchanged in 1/3.
- Action JS and device JS are worse in 3/3 seeds, although the deltas are small.
- Transition JS and TOF kept rate are mixed or near-neutral.
- Therefore SP-ST does not replicate the FR-ST action/device JS improvement. It shows a stable prior/evidence benefit, but not stable target-distribution alignment improvement.

## SP-ST Stage 3B Multi-Seed Results

Delta means are enhanced minus original:

| Metric | Mean delta | Std |
| --- | ---: | ---: |
| precision | -0.1948 | 0.2452 |
| recall | -0.1424 | 0.3446 |
| F1 | -0.1775 | 0.3141 |
| FPR | +0.0247 | 0.0528 |
| accuracy | -0.0836 | 0.1951 |

Per-seed AD deltas:

| Seed | Precision | Recall | F1 | FPR | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2024 | -0.4603 | -0.5206 | -0.5269 | +0.0838 | -0.3022 |
| 2025 | +0.0232 | +0.1538 | +0.0813 | +0.0082 | +0.0728 |
| 2026 | -0.1472 | -0.0604 | -0.0869 | -0.0179 | -0.0213 |

Interpretation:

- Enhanced does not improve SP-ST AD sanity check on average: mean F1 and accuracy decrease, and mean FPR increases.
- The result is seed-sensitive: seed 2025 improves F1/accuracy, seed 2024 strongly degrades, and seed 2026 mildly degrades while lowering FPR.
- SP-ST should be reported as mixed/negative for downstream AD under this Codex-calibrated setting, not as confirming the FR-ST AD gain.

## SP-ST Causal Edge Diagnostic

Diagnostic outputs:

```text
outputs/gcad_gss/sp_st_causal_edge_diagnostic.json
outputs/gcad_gss/sp_st_causal_edge_diagnostic.md
outputs/gcad_gss/sp_st_causal_edge_diagnostic_edges.csv
```

Main finding:

- The SP-ST GCAD prior is dominated by edges pointing to `Television`.
- In the SP spring target data, `Television` device frequency is low, about `0.0315`.
- Original codex-calibrated SP-ST TOF already overuses `Television`, about `0.1692` mean device frequency.
- Enhanced increases it further to about `0.1835`, producing a mean device gap increase of `+0.0143`.
- The corresponding action-level harms are also mostly TV-related, especially `Television:audioMute mute`, `Television:setAmbientOn`, `Television:audioMute unmute`, and `Television:volumeDown`.

Top learned SP device edges from seed 2024:

| Rank | Source | Target | Weight |
| ---: | --- | --- | ---: |
| 1 | Other | Television | 0.000181 |
| 2 | SmartLock | Television | 0.000163 |
| 3 | AirConditioner | Television | 0.000138 |
| 4 | GarageDoor | Television | 0.000109 |
| 5 | Fan | Television | 0.000099 |
| 6 | Camera | Television | 0.000099 |
| 7 | SmartPlug | Television | 0.000083 |
| 8 | Refrigerator | Television | 0.000077 |
| 9 | NetworkAudio | Television | 0.000074 |
| 10 | Dryer | Television | 0.000071 |

Interpretation:

- The GCAD prior improves low-evidence and coverage because it introduces consistent temporal structure.
- However, the dominant target endpoint is not target-context aligned for SP spring.
- Smaller `top_k` alone is unlikely to fix SP-ST, because top 10 edges are still mostly `* -> Television`.
- The next SP-ST ablation should add a target-distribution guard: suppress or downweight edges whose target endpoint is already overrepresented in the original prompt arm relative to the SP spring target.

## SP-ST Guarded-Edge Ablation

Guarded-edge ablation suppresses causal edges whose target endpoint is already overrepresented in the original prompt arm relative to the SP spring target distribution. For SP-ST this removes the dominant `* -> Television` edges and keeps lower-risk edges such as `* -> Other`.

Stage 3A guarded outputs:

```text
outputs/gcad_gss/sp_st_guarded_edge_ablation_stage3a_summary.csv
outputs/gcad_gss/sp_st_guarded_edge_ablation_stage3a_summary.json
outputs/gcad_gss/sp_st_guarded_edge_ablation_stage3a_summary.md
```

Stage 3A delta means versus original:

| Metric | Unguarded mean | Guarded mean | Guarded - unguarded |
| --- | ---: | ---: | ---: |
| low evidence rate | -0.0746 | -0.0491 | +0.0255 |
| action JS to target | +0.0096 | -0.0091 | -0.0187 |
| device JS to target | +0.0078 | -0.0122 | -0.0200 |
| causal coverage | +0.0502 | +0.0314 | -0.0188 |
| violation rate | -0.0012 | +0.0107 | +0.0119 |
| transition JS to target | +0.0020 | +0.0005 | -0.0014 |
| TOF kept rate | -0.0079 | -0.0159 | -0.0079 |

Interpretation:

- Guarding fixes the main SP-ST distribution drift: action JS and device JS become improvements on average.
- The device JS improvement is stable in 3/3 seeds.
- This costs some low-evidence and causal-coverage gains and worsens violation rate.
- The ablation confirms the diagnostic: the unguarded SP-ST failure is largely caused by overrepresented `Television`-target causal hints.

Stage 3B guarded outputs:

```text
outputs/gcad_gss/sp_st_guarded_edge_ablation_stage3b_summary.csv
outputs/gcad_gss/sp_st_guarded_edge_ablation_stage3b_summary.json
outputs/gcad_gss/sp_st_guarded_edge_ablation_stage3b_summary.md
```

Stage 3B delta means versus original:

| Metric | Unguarded mean | Guarded mean |
| --- | ---: | ---: |
| precision | -0.1948 | +0.0188 |
| recall | -0.1424 | +0.0856 |
| F1 | -0.1775 | +0.0494 |
| FPR | +0.0247 | +0.0096 |
| accuracy | -0.0836 | +0.0380 |

Per-seed guarded AD deltas:

| Seed | F1 | FPR | Accuracy | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2024 | +0.0041 | +0.0742 | -0.0137 | -0.0503 | +0.0467 |
| 2025 | +0.1161 | -0.0316 | +0.1113 | +0.0559 | +0.1909 |
| 2026 | +0.0279 | -0.0137 | +0.0165 | +0.0507 | +0.0192 |

Interpretation:

- Guarded SP-ST reverses the unguarded downstream AD result: mean F1 and accuracy become positive, and F1 is positive in 3/3 seeds.
- FPR remains mixed: seeds 2025 and 2026 improve, but seed 2024 worsens, leaving a small positive mean FPR delta.
- This is a better SP-ST candidate than unguarded GCAD-GSS, but the violation-rate and FPR tradeoffs should be reported clearly.

## SP-ST Downweighted-Edge Ablation

Downweighted-edge ablation keeps all causal edges, but multiplies overrepresented target endpoint edges by `0.25` and reorders by adjusted weight. For SP-ST, this keeps `* -> Television` edges in the prompt/generator path rather than removing them.

Stage 3A downweighted outputs:

```text
outputs/gcad_gss/sp_st_downweighted_edge_ablation_stage3a_summary.csv
outputs/gcad_gss/sp_st_downweighted_edge_ablation_stage3a_summary.json
outputs/gcad_gss/sp_st_downweighted_edge_ablation_stage3a_summary.md
```

Stage 3A delta means versus original:

| Metric | Unguarded mean | Hard guard mean | Downweighted mean |
| --- | ---: | ---: | ---: |
| low evidence rate | -0.0746 | -0.0491 | -0.0219 |
| action JS to target | +0.0096 | -0.0091 | +0.0105 |
| device JS to target | +0.0078 | -0.0122 | -0.0045 |
| causal coverage | +0.0502 | +0.0314 | -0.0016 |
| violation rate | -0.0012 | +0.0107 | -0.0011 |
| transition JS to target | +0.0020 | +0.0005 | -0.0026 |
| TOF kept rate | -0.0079 | -0.0159 | -0.0238 |

Interpretation:

- Downweighting improves device JS and transition JS relative to unguarded, but it does not fix action JS.
- It preserves violation-rate behavior better than hard guard, but loses most causal coverage and low-evidence gains.
- At factor `0.25`, downweighting is not as attractive as hard guard for Stage 3A because it keeps too much `Television` action pressure.

Stage 3B downweighted outputs:

```text
outputs/gcad_gss/sp_st_downweighted_edge_ablation_stage3b_summary.csv
outputs/gcad_gss/sp_st_downweighted_edge_ablation_stage3b_summary.json
outputs/gcad_gss/sp_st_downweighted_edge_ablation_stage3b_summary.md
```

Stage 3B delta means versus original:

| Metric | Unguarded mean | Hard guard mean | Downweighted mean |
| --- | ---: | ---: | ---: |
| precision | -0.1948 | +0.0188 | +0.0076 |
| recall | -0.1424 | +0.0856 | +0.0994 |
| F1 | -0.1775 | +0.0494 | +0.0695 |
| FPR | +0.0247 | +0.0096 | +0.1722 |
| accuracy | -0.0836 | +0.0380 | -0.0364 |

Per-seed downweighted AD deltas:

| Seed | F1 | FPR | Accuracy | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2024 | +0.0023 | +0.0591 | -0.0117 | -0.0415 | +0.0357 |
| 2025 | +0.0011 | -0.0027 | +0.0014 | +0.0021 | +0.0000 |
| 2026 | +0.2053 | +0.4602 | -0.0989 | +0.0624 | +0.2624 |

Interpretation:

- Downweighting gives positive F1 in 3/3 seeds, but this is recall-driven and comes with a large FPR spike in seed 2026.
- Mean accuracy is negative, unlike hard guard.
- Under the current factor `0.25`, hard guard is the cleaner SP-ST candidate. A stronger downweight factor, for example `0.10`, could be tested if preserving a weak TV causal signal is still desired.

## Cautious Claim

Recommended wording:

```text
On FR-ST, under a controlled Codex-calibrated GPT-style generation backend with unchanged SmartGen TOF and Transformer Autoencoder evaluation, the GCAD-GSS enhanced prompt consistently improves low-evidence rate and action/device distribution alignment across three seeds. The downstream AD sanity check is positive on average but seed-sensitive, so it should be reported as encouraging rather than robust.

On SP-ST, under the same controlled Codex-calibrated backend and unchanged SmartGen TOF/AD workflow, the GCAD-GSS enhanced prompt consistently improves causal coverage and low-evidence rate, but it does not improve action/device JS and does not improve downstream AD on average. SP-ST is seed-sensitive and should be treated as a mixed result.

A target-distribution guarded SP-ST variant suppressing overrepresented causal endpoints, especially `* -> Television`, improves action/device JS on average and restores positive downstream F1 across all three seeds, though FPR and violation-rate tradeoffs remain.

A softer downweighted SP-ST variant with factor `0.25` also restores positive F1, but it leaves action JS unimproved and causes a larger FPR/accuracy tradeoff. It is therefore weaker than hard guard under the current settings.
```

Avoid claiming:

- GCAD-GSS universally improves downstream AD.
- The FR-ST result is already validated across all SmartGen settings.
- SP-ST confirms downstream AD improvement.
- The `codex-calibrated` setting is equivalent to a fully independent external LLM/API run.

## Next Work

1. Keep hard guarded-edge SP-ST as the current best candidate; optionally test stronger downweighting such as factor `0.10` only if retaining weak `Television` causal hints is important.
2. Consider more FR-ST or SP-ST seeds if the downstream AD claim needs to be stronger than "encouraging but seed-sensitive".
3. Reuse the saved original prompt arm when the generator protocol, seed, TOF, target test, and AD settings are unchanged.
4. Rerun the original prompt arm only when changing generator protocol, seed set, data split, or evaluation settings.
5. Compare the Codex-calibrated GCAD-GSS results against the SmartGen paper table as the reported original SmartGen baseline, without rerunning the paper baseline unless a reviewer explicitly requires local reproduction.
