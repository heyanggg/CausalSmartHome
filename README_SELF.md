# CausalSmartHome 内部说明

## 当前实验口径

主实验矩阵是 FR/SP/US x ST/TT/NT，共 9 组，每组 seeds = 2024, 2025, 2026。

主实验 baseline 是 `original_gen_reference`，即 SmartGen paper Table 3 的
SmartGen 列。该 reference 位于：

```text
causal_smart_home/resources/reference/smartgen_table3_ad.json
```

这些值是 paper-reported reference baseline，不是本项目重新跑出的结果。以后如果
完整复现原 Gen，可以新增 `original_gen_rerun`，但不能混淆 reference 和 rerun。

主方法是 `proposed_causal_gss_gpt55_causal_tof`：

```text
causal prior
-> target distribution guard
-> causal-reweighted GSS
-> GPT-5.5 generation package / validated generation
-> Gen original two-stage TOF
-> Causal-TOF
-> Gen downstream AD
```

`ablation_no_causal_tof` 只用于分析 Causal-TOF 的效果。它不是原 Gen baseline，
不能出现在 main baseline 位置。

## 当前完成状态

当前 completed subset 是 SP-ST seeds 2024/2025/2026。已有历史 frozen 包：

```text
outputs/main_experiment_frozen/sp_st_gpt55_proposed_3seed_20260623
```

该包保留为历史单场景结果，不再代表完整主实验。

Gen/SmartGen 实验数据源已定位为：

```text
/home/heyang/projects/SmartGen
```

已迁移 normal pkl、downstream test/attack pkl、dictionary、models1.py，以及
FR/SP/US 的 SPPC checkpoint。迁移 manifest：

```text
outputs/data_migration/gen_data_migration_manifest.json
outputs/data_migration/gen_data_migration_report.md
```

迁移逻辑会识别这些别名，并在 manifest 中记录 `alias_used`、
`source_context_resolved` 和 `target_context`：

- context: `night`/`nighttime`, `multi`/`multiple`
- filename: `train.pkl`/`trn.pkl`, `val.pkl`/`validation.pkl`/`rs_vld.pkl`/`vld.pkl`

当前迁移后状态：

- `COMPLETE`: SP-ST seeds 2024/2025/2026
- `GENERATION_MISSING`: FR-ST, FR-TT, FR-NT, SP-TT, SP-NT, US-ST, US-TT, US-NT
- `MISSING_DATA`: none

上一轮深搜结果：

- `us/spring/vld.pkl` 使用源别名 `SmartGen/IoT_data/us/spring/rs_vld.pkl`，已迁移。
- `us/nighttime/vld.pkl` 使用源别名 `SmartGen/IoT_data/us/night/rs_vld.pkl`，已迁移。
- `sp/multiple/vld.pkl` 在 `SmartGen/IoT_data/sp/multiple/vld.pkl`，已迁移。
- `sp/multiple/trn.pkl`、`us/multiple/trn.pkl`、`us/multiple/vld.pkl` 在 SmartGen
  中没有原始 normal split；只找到 filter_data/generated 候选。

SmartGen 原始 anomaly-detection 代码对 `multiple` 有特殊口径：

```text
if new_env == 'multiple':
    vld_file = data_file
    train_file = data_file
```

因此 NT 组合的主实验数据就绪不要求 `multiple/trn.pkl` 或 `multiple/vld.pkl`；
package 构建需要 source `single/trn.pkl`、target `multiple/test.pkl`、
target `multiple/split_test.pkl`、downstream attack/test、checkpoint 和 dictionary。
不能把 filter_data/generated 候选复制成 normal input。

状态报告命令：

```bash
PYTHONPATH=. python scripts/run_main_experiment_matrix.py --dry-run --matrix all
```

输出：

```text
outputs/main_experiment/summary/matrix_status_report.md
outputs/main_experiment/summary/matrix_status_report.json
```

## 当前结果目录

历史 SP-ST 输出仍在旧布局中：

```text
outputs/main_experiment/
  gpt55_generation/sp_st/seed2024..2026
  gen_original_tof/sp_st/seed2024..2026
  causal_tof/sp_st/seed2024..2026
  downstream_ad/sp_st/seed2024..2026
```

新矩阵脚本支持清晰的 dataset/scenario/seed 布局，并会兼容读取旧 SP-ST 路径。

GPT-5.5 package 输出在：

```text
outputs/main_experiment/gpt55_generation_packages/<dataset>/<scenario>/seed<seed>/
outputs/main_experiment/gpt55_generation_packages/generation_package_index.json
outputs/main_experiment/gpt55_generation_packages/generation_package_index.md
outputs/main_experiment/gpt55_generation_packages/README_SEND_TO_GPT55.md
```

当前 package ready 共 27 个：FR/SP/US x ST/TT/NT x seeds 2024/2025/2026。

## Summary 口径

重新生成 summary：

```bash
PYTHONPATH=. python scripts/summarize_main_experiment.py --matrix all
PYTHONPATH=. python scripts/summarize_main_experiment.py --matrix all --ablation
```

输出文件：

- `main_comparison_per_seed.*`
- `main_comparison_vs_gen.*`
- `main_comparison_aggregate.*`
- `ablation_causal_tof.*`

`main_comparison_*` 只比较 `original_gen_reference` 和
`proposed_causal_gss_gpt55_causal_tof`。`ablation_no_causal_tof` 只出现在
`ablation_causal_tof.*`。

## 补齐缺失组合流程

对每个缺失 dataset-scenario-seed：

1. 对 `MISSING_DATA` 组合，先补齐 manifest 中列出的 normal pkl。
2. 运行 `scripts/build_causal_gss_prompt.py` 生成 causal prior、guard report、reweighted GSS hints 和 prompt。
3. 运行 `scripts/run_main_experiment_matrix.py --stage build_generation_package --matrix all` 生成 GPT-5.5 package。
4. 补入 GPT-5.5 JSONL 后，运行 `scripts/run_main_experiment_matrix.py --stage validate_generation --matrix all`。
5. 对已有 validated generation pkl，运行 `scripts/run_main_experiment_matrix.py --stage downstream --matrix all`。
6. 重新运行 summary 和 freeze。

如果某组合缺少数据或输出，脚本必须标记明确状态，例如 `MISSING_DATA`、
`GENERATION_MISSING` 或 `MISSING_DOWNSTREAM_RESULT`，不能补假结果。

注意：package 是 prompt/schema，不是 GPT-5.5 已生成结果。GPT-5.5 JSONL 应放在：

```text
outputs/main_experiment/gpt55_generation/<dataset>_<scenario>/seed<seed>/generated_gpt55_clean.jsonl
```

## 冻结

冻结完整矩阵状态：

```bash
PYTHONPATH=. python scripts/freeze_main_experiment.py --matrix all
```

新包名形如：

```text
outputs/main_experiment_frozen/main_matrix_fr_sp_us_st_tt_nt_gpt55_3seed_YYYYMMDD
```

manifest 会包含 experiment matrix、per dataset/scenario/seed status、各阶段路径、
summary 路径和 original Gen reference JSON。
