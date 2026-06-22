# Task 16: Stage4 SmartGen Built-in Downstream AD

Date: 2026-06-22

This note corrects the Stage4 downstream AD mainline:

```text
Main downstream AD = SmartGen built-in anomaly_detection_pipeline semantics
SmartGuard AD = optional extension only
```

The previous `epochs=1` SmartGuard Stage4B runs are smoke tests. They are not used in the main result table.

## Protocol

Wrapper:

```text
scripts/run_stage4c_gen_builtin_downstream_ad.py
```

The wrapper reuses `causal_smart_home.smartgen_experiment.run_smartgen_anomaly_experiment`, which imports SmartGen `anomaly_detection_pipeline/models1.py` and runs the SmartGen TransformerAutoencoder reconstruction-loss workflow. It does not modify SmartGen, SmartGuard, or GCAD core code.

Inputs:

```text
Stage3 baseline: matching Stage3 original-prompt smartgen_tof.pkl
Stage4 raw: generated_codex_gpt55_clean.pkl
Stage4 Causal-TOF: generated_weighted_resampled.pkl
```

Common settings:

```text
datasets = fr_st, sp_st
seeds = 2024, 2025, 2026
epochs = 15
split_ratio = 0.8
threshold = SmartGen validation-loss percentile
FR-ST percentile = 95.5
SP-ST percentile = 95.0
api_llm = false
```

The actual runtime device recorded by the wrapper was `cpu` for all runs.

## Outputs

Entry/protocol reports:

```text
outputs/gcad_gss_stage4/smartgen_downstream_ad_entry_report.md
outputs/gcad_gss_stage4/stage3_downstream_ad_protocol_report.md
outputs/gcad_gss_stage4/gen_builtin_ad_config.json
```

Run outputs:

```text
outputs/gcad_gss_stage4/gen_builtin_ad/fr_st/*/downstream_ad_metrics.json
outputs/gcad_gss_stage4/gen_builtin_ad/sp_st/*/downstream_ad_metrics.json
```

Summary:

```text
outputs/gcad_gss_stage4/gen_builtin_ad_summary.csv
outputs/gcad_gss_stage4/gen_builtin_ad_summary.md
outputs/gcad_gss_stage4/gen_builtin_ad_summary.json
```

All 18 formal SmartGen built-in AD runs succeeded. No failure report was recorded.

## Results

Mean over three seeds:

| Dataset | Variant | F1 | FPR | Accuracy | Successful seeds |
| --- | --- | ---: | ---: | ---: | ---: |
| FR-ST | Stage3 prompt-only baseline | 0.8175 | 0.4470 | 0.7765 | 3 |
| FR-ST | Stage4 downweight raw | 0.2439 | 0.2273 | 0.4716 | 3 |
| FR-ST | Stage4 downweight + Causal-TOF resampled | 0.4442 | 0.2273 | 0.5928 | 3 |
| SP-ST | Stage3 prompt-only baseline | 0.5461 | 0.2216 | 0.6472 | 3 |
| SP-ST | Stage4 downweight raw | 0.8884 | 0.0929 | 0.8924 | 3 |
| SP-ST | Stage4 downweight + Causal-TOF resampled | 0.8347 | 0.1410 | 0.8384 | 3 |

Key deltas:

| Dataset | Comparison | Delta F1 | Delta FPR | Delta accuracy |
| --- | --- | ---: | ---: | ---: |
| FR-ST | Stage4 raw vs Stage3 | -0.5736 | -0.2197 | -0.3049 |
| FR-ST | Stage4 TOF vs Stage3 | -0.3732 | -0.2197 | -0.1837 |
| FR-ST | Stage4 TOF vs Stage4 raw | +0.2003 | +0.0000 | +0.1212 |
| SP-ST | Stage4 raw vs Stage3 | +0.3423 | -0.1287 | +0.2452 |
| SP-ST | Stage4 TOF vs Stage3 | +0.2885 | -0.0806 | +0.1912 |
| SP-ST | Stage4 TOF vs Stage4 raw | -0.0537 | +0.0481 | -0.0540 |

## Interpretation

FR-ST does not support an AD improvement claim. Stage4 raw and Stage4 Causal-TOF resampled both underperform the Stage3 prompt-only baseline in mean F1 and accuracy, although FPR is lower. Causal-TOF improves over Stage4 raw on FR-ST, but not enough to beat Stage3.

SP-ST supports a preliminary Stage4 AD improvement claim. Stage4 raw improves mean F1 and accuracy over Stage3 and lowers mean FPR across three seeds. Stage4 Causal-TOF resampled also improves over Stage3, but it is worse than Stage4 raw on mean F1, FPR, and accuracy.

Robust improvement across both scenarios cannot be claimed, because FR-ST is negative. The defensible conclusion is partial/preliminary: SP-ST improves under SmartGen built-in AD; FR-ST does not.

Suppress-mode Stage4 remains a negative/control path, not the downweight mainline.
