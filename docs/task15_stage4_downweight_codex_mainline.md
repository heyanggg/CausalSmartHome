# Task 15: Stage4 Downweight Codex/GPT-5.5 Mainline

Date: 2026-06-22

This note supersedes the earlier suppress-mode Stage4 default for the current mainline. The fixed Stage4 mainline is:

```text
generator = codex_gpt55_surrogate
api_llm = false
surrogate_for_smartgen_llm = true
guard-mode = downweight
downweight-factor = 0.25
reweight-mode = multiplicative
lambda-causal = 1.0
max-overuse-ratio = 1.25
endpoint-policy = target
```

No SmartGen, SmartGuard, or GCAD core source was modified.

## 1. Fresh Surrogate Generation

Generation packages:

```text
outputs/gcad_gss_stage4/codex_gpt55_generation/fr_st_seed2024
outputs/gcad_gss_stage4/codex_gpt55_generation/sp_st_seed2024
```

Each package contains `prompt.txt`, `guard_report.json`, `guarded_reweighted_gss_hints.json`, `resolved_gcad_prior.json`, `generation_schema.json`, `generation_instruction.md`, `generated_raw.jsonl`, `generated_codex_gpt55.pkl`, `generated_codex_gpt55_clean.pkl`, `generation_metadata.json`, and validation reports.

Validation passed without cleaning loss:

| Scenario | Raw | Clean | Invalid | Status |
| --- | ---: | ---: | ---: | --- |
| FR-ST | 209 | 209 | 0 | valid |
| SP-ST | 36 | 36 | 0 | valid |

The generator was tightened to enforce device-action compatibility after an initial FR-ST validation found `Other(18)` paired with `None:refresh(96)`. The final regenerated pkl files have no invalid sequences.

## 2. Stage4A Quality

Main output directories:

```text
outputs/gcad_gss_stage4/fr_st_downweight_multiplicative_codex_gpt55_seed2024
outputs/gcad_gss_stage4/sp_st_downweight_multiplicative_codex_gpt55_seed2024
```

Key metrics:

| Scenario | Generated | Action JS | Device JS | Transition JS | Causal coverage | Violation | Low evidence | Nonzero edges | Downweighted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FR-ST | 209 | 0.4351 | 0.1261 | 0.3604 | 0.9161 | 0.0839 | 0.6842 | 5 | 15 |
| SP-ST | 36 | 0.3561 | 0.1831 | 0.3516 | 1.0000 | 0.0000 | 1.0000 | 9 | 48 |

Television diagnostics:

| Scenario | TV key | Generated freq | Target freq | Ratio |
| --- | --- | ---: | ---: | ---: |
| FR-ST | `d:29` | 0.0897 | 0.1056 | 0.8493 |
| SP-ST | `d:30` | 0.0000 | 0.0315 | 0.0000 |

SP-ST Television overuse is controlled in the fresh Codex/GPT-5.5 surrogate output.

## 3. Causal-TOF

Causal-TOF was run as soft weighting/resampling, not hard deletion.

| Scenario | Num seq | Avg coverage | Avg violation | Avg dist penalty | Avg final score | Avg weight | Min weight | Max weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FR-ST | 209 | 0.9161 | 0.0839 | 0.4512 | 0.5351 | 0.3858 | 0.0381 | 0.6524 |
| SP-ST | 36 | 1.0000 | 0.0000 | 0.4605 | 0.4605 | 0.4291 | 0.1662 | 0.6461 |

Per-directory summaries are saved as `causal_tof_summary.json`.

## 4. Downstream AD

Real SmartGuard wrapper attempts were completed for the four requested Stage4B directories using `epochs=1`, `sequence-length=40`, and `seed=2024`. These are interface-level smoke runs, not fully trained downstream claims.

| Scenario | Variant | F1 | FPR | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| FR-ST | raw | 0.2409 | 0.0546 | 0.8436 | 0.1405 |
| FR-ST | TOF weighted/resampled | 0.2453 | 0.0562 | 0.8426 | 0.1435 |
| SP-ST | raw | 0.3195 | 0.0572 | 0.8847 | 0.1949 |
| SP-ST | TOF weighted/resampled | 0.3220 | 0.0572 | 0.8856 | 0.1968 |

The normalized files are:

```text
outputs/gcad_gss_stage4/fr_st_stage4b_ad_codex_gpt55_downweight_raw_seed2024/downstream_ad_metrics.json
outputs/gcad_gss_stage4/fr_st_stage4b_ad_codex_gpt55_downweight_tof_seed2024/downstream_ad_metrics.json
outputs/gcad_gss_stage4/sp_st_stage4b_ad_codex_gpt55_downweight_raw_seed2024/downstream_ad_metrics.json
outputs/gcad_gss_stage4/sp_st_stage4b_ad_codex_gpt55_downweight_tof_seed2024/downstream_ad_metrics.json
```

Do not claim robust downstream AD improvement from these smoke metrics. They show that the SmartGuard integration path is executable for the Stage4 Codex/GPT-5.5 artifacts.

## 5. Answers

1. Fresh Codex/GPT-5.5 surrogate pkl exists: yes, for FR-ST and SP-ST seed 2024.
2. API LLM was used: no. Metadata records `api_llm=false`.
3. SmartGen/SmartGuard/GCAD core was modified: no.
4. Mainline guard is suppress: no. Mainline is downweight with factor 0.25.
5. Reweighting mode is additive: no. Mainline is multiplicative; additive remains ablation only.
6. SP-ST Television bias is still overgenerated: no in the fresh surrogate; generated frequency is 0.
7. Validation found illegal sequences: no in the final pkl.
8. Causal-TOF hard-deleted samples: no. It produced weights and weighted-resampled pkl.
9. Real downstream AD completed: yes as SmartGuard `epochs=1` smoke runs.
10. Robust AD lift can be claimed: no. Full training and baseline-controlled repeats are still required.

## 6. Summary Outputs

```text
outputs/gcad_gss_stage4/stage4_downweight_codex_gpt55_summary.csv
outputs/gcad_gss_stage4/stage4_downweight_codex_gpt55_summary.md
outputs/gcad_gss_stage4/stage4_downstream_ad_summary.csv
outputs/gcad_gss_stage4/stage4_downstream_ad_summary.md
```

