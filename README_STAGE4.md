# Stage 4 GCAD-to-SmartGen Glue 使用说明

本次修改只新增 Stage 4 glue 层，不删除旧 Stage 3 脚本，不覆盖 `outputs/gcad_gss/`。

## 快速验证

```bash
cd /home/heyang/projects/CausalSmartHome
PYTHONPATH=. pytest -q tests
```

## 构建 guarded causal-reweighted GSS prompt

```bash
python scripts/build_guarded_causal_reweighted_gss_prompt.py \
  --source-pkl path/to/source_normal.pkl \
  --target-pkl path/to/target_normal.pkl \
  --prior-json path/to/causal_prior.json \
  --device-dict path/to/dictionary.py \
  --out-prompt outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024/prompt.txt \
  --out-prior-json outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024/resolved_gcad_prior.json \
  --out-guard-report outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024/guard_report.json \
  --out-reweighted-hints outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024/guarded_reweighted_gss_hints.json \
  --lambda-causal 1.0 \
  --reweight-mode multiplicative \
  --guard-mode suppress \
  --endpoint-policy target \
  --max-overuse-ratio 1.25 \
  --top-k 50 \
  --seed 2024
```

若不传 `--prior-json`，脚本会通过 `gcad_prior_source.resolve_gcad_prior()` 调用当前已有 `GCADAdapter.mine_event_prior()`；输出会明确标注 `gcad_source=existing_adapter_compact_fallback`，不能把它写成官方 GCAD reproduction。

## Causal-TOF soft weighting

```bash
python scripts/run_causal_tof_weighting.py \
  --generated-pkl path/to/generated.pkl \
  --guarded-hints-json outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024/guarded_reweighted_gss_hints.json \
  --target-pkl path/to/target_normal.pkl \
  --out-scores outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024/causal_tof_scores.json \
  --out-weights outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024/generated.weights.json \
  --out-weighted-resampled-pkl outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024/generated_weighted_resampled.pkl \
  --mode weight \
  --temperature 2.0 \
  --seed 2024
```

## Verifier + repair prompt

```bash
python scripts/verify_and_repair_generated_sequences.py \
  --generated-pkl path/to/generated.pkl \
  --guarded-hints-json outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024/guarded_reweighted_gss_hints.json \
  --guard-report-json outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024/guard_report.json \
  --target-pkl path/to/target_normal.pkl \
  --out-dir outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024/repair
```

## 汇总

```bash
python scripts/summarize_stage4_guarded_reweighted.py
python scripts/summarize_stage4_causal_tof.py
```

详情见 `docs/task13_deeper_gcad_smartgen_integration.md`。

## 2026-06-22 server run status

Server-side Stage 4 validation was run under:

```bash
cd /home/heyang/projects/CausalSmartHome
PYTHONPATH=. pytest -q tests
```

Current result:

```text
26 passed, 5 skipped
```

Real FR-ST and SP-ST inputs were found and used for Stage 4A:

```text
FR source: /home/heyang/projects/SmartGen/behavior_prediciton_baseline/SASRec/baseline_data/fr/winter/trn.pkl
FR target: /home/heyang/projects/SmartGen/parameter_study/test/fr/spring/split_test.pkl
SP source: /home/heyang/projects/SmartGen/behavior_prediciton_baseline/SASRec/baseline_data/sp/daytime/trn.pkl
SP target: /home/heyang/projects/SmartGen/parameter_study/test/sp/spring/split_test.pkl
Generated inputs: outputs/gcad_gss/*/smartgen_tof.pkl from the Stage 3 codex-calibrated enhanced arms
Prior inputs: outputs/gcad_gss/*/quality_eval/causal_prior_source.json from the matching Stage 3 runs
Device mapping: /home/heyang/projects/SmartGen/SmartGen/dictionary.py
```

Primary Stage 4 outputs are:

```text
outputs/gcad_gss_stage4/fr_st_guarded_reweighted_seed2024
outputs/gcad_gss_stage4/fr_st_guarded_reweighted_seed2025
outputs/gcad_gss_stage4/fr_st_guarded_reweighted_seed2026
outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024
outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2025
outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2026
```

The target-distribution guard correctly flags the SP-ST Television bias. In
`sp_st_guarded_reweighted_seed2024/guard_report.json`, `AirConditioner -> Television`
is suppressed because `Television` has observed source frequency `0.668817` versus target
frequency `0.031471`, ratio `21.252`.

Important caveat: with the default `guard-mode=suppress`, all top FR-ST and SP-ST causal
edges were suppressed in the primary runs (`guarded_edge_count=0`). This means the current
primary artifacts validate the guard and prompt/report plumbing, but the effective causal
reweight contribution is zero after guarding. Extra seed-2024 downweight ablations were run
to exercise nonzero causal reweighting:

```text
outputs/gcad_gss_stage4/fr_st_downweight_multiplicative_seed2024
outputs/gcad_gss_stage4/fr_st_downweight_additive_seed2024
outputs/gcad_gss_stage4/sp_st_downweight_multiplicative_seed2024
outputs/gcad_gss_stage4/sp_st_downweight_additive_seed2024
```

Stage 4B downstream AD was not truly completed. The Stage 4B scripts wrote transparent
`--dry-run` records under `outputs/gcad_gss_stage4/*stage4b_ad*` because no real SmartGuard
downstream training/evaluation metrics were produced for the new Stage 4 artifacts in this
run. Do not report an AD lift from Stage 4 yet.

Full details are in:

```text
docs/task14_stage4_server_run_report.md
outputs/gcad_gss_stage4/stage4_guarded_reweighted_summary.md
outputs/gcad_gss_stage4/stage4_causal_tof_summary.md
```
