# Task 14: Stage 4 Server Run Report

Date: 2026-06-22

Repo:

```text
/home/heyang/projects/CausalSmartHome
```

Git commit at run start:

```text
56138c64cb83f1c47680e71aaf1b985a021698d7
```

## 1. Environment And Git Status

External projects exist:

```text
/home/heyang/projects/SmartGen
/home/heyang/projects/SmartGuard
/home/heyang/projects/GCAD
```

`external_sources/SmartGen`, `external_sources/SmartGuard`, and `external_sources/GCAD` resolve to
those real project paths.

`git status --short` showed Stage 4 files as untracked plus an existing modification to
`causal_smart_home/causal_prior.py`. These were treated as the current Stage 4 patch state and were
not reverted.

## 2. Patch Acceptance

All requested Stage 4 modules and scripts were present:

```text
causal_smart_home/gcad_prior_source.py
causal_smart_home/target_distribution_guard.py
causal_smart_home/causal_gss_reweight.py
causal_smart_home/causal_tof.py
scripts/build_guarded_causal_reweighted_gss_prompt.py
scripts/run_causal_tof_weighting.py
scripts/verify_and_repair_generated_sequences.py
scripts/run_stage4a_guarded_reweighted_gss_fr_st.py
scripts/run_stage4a_guarded_reweighted_gss_sp_st.py
scripts/run_stage4b_ad_guarded_reweighted_gss_fr_st.py
scripts/run_stage4b_ad_guarded_reweighted_gss_sp_st.py
scripts/run_stage4b_ad_causal_tof_weighted_fr_st.py
scripts/run_stage4b_ad_causal_tof_weighted_sp_st.py
scripts/summarize_stage4_guarded_reweighted.py
scripts/summarize_stage4_causal_tof.py
README_STAGE4.md
docs/task13_deeper_gcad_smartgen_integration.md
```

## 3. Test Results

Initial test run:

```text
24 passed, 5 skipped
```

After glue fixes:

```text
PYTHONPATH=. pytest -q tests
26 passed, 5 skipped
```

Results were saved to:

```text
outputs/gcad_gss_stage4/test_results_after_codex.txt
outputs/gcad_gss_stage4/final_test_results.txt
```

## 4. Real Data Paths

Pickle format checks confirmed list-of-flat-four-tuples, e.g.
`[day, hour_slot, device_id, action_id, ...]`.

| Scenario | Source normal pkl | Target normal pkl |
| --- | --- | --- |
| FR-ST | `/home/heyang/projects/SmartGen/behavior_prediciton_baseline/SASRec/baseline_data/fr/winter/trn.pkl` | `/home/heyang/projects/SmartGen/parameter_study/test/fr/spring/split_test.pkl` |
| SP-ST | `/home/heyang/projects/SmartGen/behavior_prediciton_baseline/SASRec/baseline_data/sp/daytime/trn.pkl` | `/home/heyang/projects/SmartGen/parameter_study/test/sp/spring/split_test.pkl` |

Generated and prior inputs came from the existing Stage 3 enhanced arms:

| Scenario | Seed | Generated pkl | Prior JSON |
| --- | ---: | --- | --- |
| FR-ST | 2024 | `outputs/gcad_gss/fr_st_codex_calibrated_v3/fr_st_enhanced/smartgen_tof.pkl` | `outputs/gcad_gss/fr_st_codex_calibrated_v3/fr_st_enhanced/quality_eval/causal_prior_source.json` |
| FR-ST | 2025 | `outputs/gcad_gss/fr_st_codex_calibrated_seed2025/fr_st_enhanced/smartgen_tof.pkl` | `outputs/gcad_gss/fr_st_codex_calibrated_seed2025/fr_st_enhanced/quality_eval/causal_prior_source.json` |
| FR-ST | 2026 | `outputs/gcad_gss/fr_st_codex_calibrated_seed2026/fr_st_enhanced/smartgen_tof.pkl` | `outputs/gcad_gss/fr_st_codex_calibrated_seed2026/fr_st_enhanced/quality_eval/causal_prior_source.json` |
| SP-ST | 2024 | `outputs/gcad_gss/sp_st_codex_calibrated_seed2024/sp_st_enhanced/smartgen_tof.pkl` | `outputs/gcad_gss/sp_st_codex_calibrated_seed2024/sp_st_enhanced/quality_eval/causal_prior_source.json` |
| SP-ST | 2025 | `outputs/gcad_gss/sp_st_codex_calibrated_seed2025/sp_st_enhanced/smartgen_tof.pkl` | `outputs/gcad_gss/sp_st_codex_calibrated_seed2025/sp_st_enhanced/quality_eval/causal_prior_source.json` |
| SP-ST | 2026 | `outputs/gcad_gss/sp_st_codex_calibrated_seed2026/sp_st_enhanced/smartgen_tof.pkl` | `outputs/gcad_gss/sp_st_codex_calibrated_seed2026/sp_st_enhanced/quality_eval/causal_prior_source.json` |

Device names were resolved with:

```text
/home/heyang/projects/SmartGen/SmartGen/dictionary.py
```

## 5. FR-ST Stage 4A Results

Primary outputs:

```text
outputs/gcad_gss_stage4/fr_st_guarded_reweighted_seed2024
outputs/gcad_gss_stage4/fr_st_guarded_reweighted_seed2025
outputs/gcad_gss_stage4/fr_st_guarded_reweighted_seed2026
```

Each directory contains:

```text
prompt.txt
resolved_gcad_prior.json
guard_report.json
guarded_reweighted_gss_hints.json
generation_quality_metrics.json
generated.pkl
```

Main FR-ST suppress-mode metrics:

| Seed | Generated | Action JS | Device JS | Transition JS | Suppressed edges | Nonzero guarded edges |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024 | 209 | 0.6507 | 0.4618 | 0.8019 | 15 | 0 |
| 2025 | 210 | 0.6556 | 0.4549 | 0.8029 | 15 | 0 |
| 2026 | 205 | 0.6537 | 0.4620 | 0.8091 | 15 | 0 |

## 6. SP-ST Stage 4A Results

Primary outputs:

```text
outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024
outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2025
outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2026
```

Main SP-ST suppress-mode metrics:

| Seed | Generated | Action JS | Device JS | Transition JS | Suppressed edges | Nonzero guarded edges |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024 | 36 | 0.8263 | 0.6729 | 0.8824 | 48 | 0 |
| 2025 | 37 | 0.8232 | 0.6642 | 0.8718 | 48 | 0 |
| 2026 | 41 | 0.8163 | 0.6535 | 0.8650 | 48 | 0 |

SP-ST Television bias was detected. For seed 2024:

```text
AirConditioner -> Television
guard_action: suppress
reason: target endpoint Television overused: observed_freq=0.668817, target_freq=0.031471, ratio=21.252
```

## 7. Causal-TOF Results

Causal-TOF soft weighting was run for all six primary Stage 4A directories. Outputs:

```text
causal_tof_scores.json
generated.weights.json
generated_weighted_resampled.pkl
generated_weighted_resampled.pkl.config.json
```

Mean sample weights:

| Scenario | Seed 2024 | Seed 2025 | Seed 2026 |
| --- | ---: | ---: | ---: |
| FR-ST | 0.2981 | 0.3055 | 0.3026 |
| SP-ST | 0.2244 | 0.2216 | 0.2227 |

Because suppress mode left zero nonzero guarded causal edges, Causal-TOF did not measure causal-order
violations in these primary runs. The weights are driven by distribution penalty.

## 8. Repair Prompt Check

Repair prompts were generated under each primary output:

```text
outputs/gcad_gss_stage4/*_guarded_reweighted_seed*/repair/
```

The SP-ST seed 2024 prompt includes:

```text
Violated causal edges that should be repaired if possible
Overused devices from the target-distribution guard report
Television
Target context distribution warning
```

In suppress mode, violated causal edge lists are empty because all guarded causal edges were suppressed.
Prompts are still emitted because distribution penalty is positive.

## 9. Stage 4B AD Result

Stage 4B downstream AD was not truly completed. Dry-run records were written:

```text
outputs/gcad_gss_stage4/fr_st_stage4b_ad_guarded_reweighted_seed*
outputs/gcad_gss_stage4/sp_st_stage4b_ad_guarded_reweighted_seed*
outputs/gcad_gss_stage4/fr_st_stage4b_ad_causal_tof_weighted_seed*
outputs/gcad_gss_stage4/sp_st_stage4b_ad_causal_tof_weighted_seed*
```

Reason: no real SmartGuard/SmartGen downstream AD run was executed for the Stage 4 generated or
weighted/resampled pkl, and no real Stage 4 `metrics.json` or `downstream_ad_metrics.json` was
provided. Therefore F1/precision/recall/FPR are `null` and no AD improvement is claimed.

## 10. Stage 3 Comparison

Stage 3 is still the only place with real downstream AD metrics. Stage 3 showed:

- FR-ST prompt-only GCAD-GSS was positive on average but seed-sensitive.
- SP-ST unguarded prompt-only was mixed/negative for AD.
- SP-ST guarded prompt-only fixed much of the Television drift and restored positive F1 across seeds.

Stage 4 primary suppress runs confirm the guard mechanism on real FR-ST/SP-ST inputs, but they do not
yet improve beyond Stage 3 because they reused Stage 3 generated pkl and did not perform a new
generation pass from the Stage 4 prompt.

## 11. Failure Points And Fixes

Observed failure or gap:

- Running Stage 4A without `--prior-json` under the default Python failed because torch is unavailable.
- Stage 4A wrapper did not expose `--sparse-threshold`.
- Stage 4B wrappers did not accept `--stage4a-dir` and weighted wrappers did not accept `--weighted-generated-pkl`.
- Summary scripts accepted `--stage4-dir` but not the requested `--root`.
- Guard report edge rows had readable device names, but `overused_devices` summary still showed `device_30`.

Fixes made:

- Used existing Stage 3 `causal_prior_source.json` files instead of recomputing prior with torch.
- Added Stage 4A `--sparse-threshold` pass-through.
- Added Stage 4B provenance args.
- Added `--root` alias to both summary scripts.
- Added readable device names to guard-report `overused_devices`.
- Added tests for Stage 4B help args and guard-report name mapping.

## 12. Redundant File Suggestions

Do not delete:

```text
outputs/gcad_gss/
scripts/run_stage3a_*.py
scripts/run_stage3b_*.py
scripts/build_gcad_gss_prompt.py
docs/task12_gcad_gss_prompt_stage3.md
```

Can be deleted later if only real runs are needed:

```text
outputs/gcad_gss_stage4/demo_toy/
outputs/gcad_gss_stage4/demo_toy_stage4a_sp_wrapper/
outputs/gcad_gss_stage4/demo_toy_stage4b_sp_wrapper/
```

Keep for this run:

```text
outputs/gcad_gss_stage4/fr_st_guarded_reweighted_seed*
outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed*
outputs/gcad_gss_stage4/*downweight*seed2024
outputs/gcad_gss_stage4/*stage4b_ad*
outputs/gcad_gss_stage4/stage4_*_summary.*
outputs/gcad_gss_stage4/*test_results*.txt
```

## 13. Next Steps

1. Generate new FR-ST/SP-ST sequences from the Stage 4 guarded causal-reweighted prompt, not by
   reusing Stage 3 enhanced generated pkl.
2. Rerun Stage 4A metrics on those newly generated sequences.
3. Run real Stage 4B downstream AD with the Stage 4 generated pkl and with Causal-TOF weighted
   resampling.
4. Compare suppress versus downweight at factors `0.25` and `0.10`.
5. Prefer multiplicative reweighting as the safer default; additive should be treated as a stronger
   ablation because it can lift weak transition edges when causal strength is high.
6. For SP-ST, keep explicit Television diagnostics in the report.

## 14. Copyable Command List

Primary Stage 4A examples:

```bash
python scripts/run_stage4a_guarded_reweighted_gss_fr_st.py \
  --source-pkl /home/heyang/projects/SmartGen/behavior_prediciton_baseline/SASRec/baseline_data/fr/winter/trn.pkl \
  --target-pkl /home/heyang/projects/SmartGen/parameter_study/test/fr/spring/split_test.pkl \
  --prior-json outputs/gcad_gss/fr_st_codex_calibrated_v3/fr_st_enhanced/quality_eval/causal_prior_source.json \
  --generated-pkl outputs/gcad_gss/fr_st_codex_calibrated_v3/fr_st_enhanced/smartgen_tof.pkl \
  --device-dict /home/heyang/projects/SmartGen/SmartGen/dictionary.py \
  --out-dir outputs/gcad_gss_stage4/fr_st_guarded_reweighted_seed2024 \
  --seed 2024

python scripts/run_stage4a_guarded_reweighted_gss_sp_st.py \
  --source-pkl /home/heyang/projects/SmartGen/behavior_prediciton_baseline/SASRec/baseline_data/sp/daytime/trn.pkl \
  --target-pkl /home/heyang/projects/SmartGen/parameter_study/test/sp/spring/split_test.pkl \
  --prior-json outputs/gcad_gss/sp_st_codex_calibrated_seed2024/sp_st_enhanced/quality_eval/causal_prior_source.json \
  --generated-pkl outputs/gcad_gss/sp_st_codex_calibrated_seed2024/sp_st_enhanced/smartgen_tof.pkl \
  --device-dict /home/heyang/projects/SmartGen/SmartGen/dictionary.py \
  --out-dir outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024 \
  --seed 2024
```

Causal-TOF example:

```bash
python scripts/run_causal_tof_weighting.py \
  --generated-pkl outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024/generated.pkl \
  --guarded-hints-json outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024/guarded_reweighted_gss_hints.json \
  --target-pkl /home/heyang/projects/SmartGen/parameter_study/test/sp/spring/split_test.pkl \
  --out-scores outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024/causal_tof_scores.json \
  --out-weights outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024/generated.weights.json \
  --out-weighted-resampled-pkl outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024/generated_weighted_resampled.pkl \
  --mode weight \
  --temperature 2.0 \
  --seed 2024
```

Repair prompt example:

```bash
python scripts/verify_and_repair_generated_sequences.py \
  --generated-pkl outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024/generated.pkl \
  --guarded-hints-json outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024/guarded_reweighted_gss_hints.json \
  --guard-report-json outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024/guard_report.json \
  --target-pkl /home/heyang/projects/SmartGen/parameter_study/test/sp/spring/split_test.pkl \
  --out-dir outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024/repair
```

Stage 4B dry-run examples:

```bash
python scripts/run_stage4b_ad_guarded_reweighted_gss_sp_st.py \
  --stage4a-dir outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024 \
  --out-dir outputs/gcad_gss_stage4/sp_st_stage4b_ad_guarded_reweighted_seed2024 \
  --seed 2024 \
  --dry-run

python scripts/run_stage4b_ad_causal_tof_weighted_sp_st.py \
  --stage4a-dir outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024 \
  --weighted-generated-pkl outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024/generated_weighted_resampled.pkl \
  --out-dir outputs/gcad_gss_stage4/sp_st_stage4b_ad_causal_tof_weighted_seed2024 \
  --seed 2024 \
  --dry-run
```

Summaries:

```bash
python scripts/summarize_stage4_guarded_reweighted.py --root outputs/gcad_gss_stage4
python scripts/summarize_stage4_causal_tof.py --root outputs/gcad_gss_stage4
```

## 15. Downweight Codex/GPT-5.5 Mainline Follow-up

After this suppress-mode server report, the Stage 4 mainline was updated to the downweight
Codex/GPT-5.5 surrogate protocol:

```text
guard-mode=downweight
downweight-factor=0.25
reweight-mode=multiplicative
lambda-causal=1.0
endpoint-policy=target
generator=codex_gpt55_surrogate
api_llm=false
surrogate_for_smartgen_llm=true
```

Fresh seed-2024 surrogate outputs were generated, validated, passed Stage4A quality checks,
and received Causal-TOF soft weights:

```text
outputs/gcad_gss_stage4/fr_st_downweight_multiplicative_codex_gpt55_seed2024
outputs/gcad_gss_stage4/sp_st_downweight_multiplicative_codex_gpt55_seed2024
```

SmartGuard Stage4B smoke runs completed with `epochs=1` for FR/SP raw and weighted variants.
They prove the integration path executes but should not be reported as robust downstream AD
improvement.

See `docs/task15_stage4_downweight_codex_mainline.md`.
