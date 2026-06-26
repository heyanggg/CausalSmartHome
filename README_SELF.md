# CausalSmartHome 内部说明

## 主实验口径

本项目是 Gen + GCAD 的缝合项目，但已经确定为一个完整主实验流程，而不是若干临时
模块的拼接。主实验矩阵是：

```text
FR/SP/US x spring/night/multiple
```

短场景名：

- `st` = spring，源上下文是 winter。
- `tt` = night，源上下文是 daytime。
- `nt` = multiple，源上下文是 single。

完整 proposed 流程固定为：

```text
因果关系先验
-> target-distribution guard
-> causal-reweighted GSS
-> Codex 生成目标上下文正常行为序列
-> Gen 原始 two-stage TOF
-> Causal-TOF
-> Gen built-in downstream AD
-> per-seed summary
```

注意：Causal-TOF 是主实验流程中的一环，不是额外附加模块。`ablation_no_causal_tof`
只是用于说明缺少这一环时结果如何变化，不能把 Causal-TOF 写成主方法之外的东西。

新实验的 proposed variant 使用：

```text
proposed_causal_gss_codex_causal_tof
```

旧输出中的 proposed 名称由 summary 脚本兼容映射到新名称。以后文档和新结果都使用
Codex 名称。

## 结果展示规则

三 seed 结果必须逐 seed 列出来。不要把均值表作为主结果，也不要在与 Gen 原论文异常
检测结果对比时做差值表。正确做法是并排列出：

- dataset
- scenario
- seed
- Gen paper/project AD F1
- ablation_no_causal_tof F1
- proposed_causal_gss_codex_causal_tof F1
- proposed precision / recall / FPR
- device

当前已完成的 SP-ST / SP-spring 与 SP-TT / SP-night 结果：

| dataset | scenario | seed | Gen paper AD F1 | ablation_no_causal_tof F1 | proposed_causal_gss_codex_causal_tof F1 | proposed precision | proposed recall | proposed FPR | device |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sp | spring | 2024 | 0.919057 | 0.741344 | 0.965517 | 0.933333 | 1.000000 | 0.071429 | cuda |
| sp | spring | 2025 | 0.919057 | 0.974565 | 0.981132 | 0.962963 | 1.000000 | 0.038462 | cuda |
| sp | spring | 2026 | 0.919057 | 0.978665 | 0.979012 | 0.965287 | 0.993132 | 0.035714 | cuda |
| sp | night | 2024 | 0.962482 | 0.786219 | 0.962482 | 0.927678 | 1.000000 | 0.077960 | cuda |
| sp | night | 2025 | 0.962482 | 0.962482 | 0.962190 | 0.927639 | 0.999414 | 0.077960 | cuda |
| sp | night | 2026 | 0.962482 | 0.841575 | 0.962190 | 0.927639 | 0.999414 | 0.077960 | cuda |

本地结果位置：

```text
outputs/main_experiment/sp_st/
outputs/main_experiment/sp_tt/
outputs/main_experiment/summary/
```

临时诊断目录如 `*_beta0`、`*_beta005`、`*_beta01` 只用于定位问题，不能进入正式
summary。`scripts/summarize_main_experiment.py` 已改成只输出 per-seed 表，不再生成
aggregate 或 seed-delta 文件。

SP-night 不能照搬 SP-spring 的 4-event 短序列生成。2026-06-26 扩展 `sp_tt`
时先用 16-int 固定短序列，downstream AD 退化到 F1 约 0.72；改为匹配
SP-night / Gen 原 synthetic 的 5-13 event 变长正常行为后恢复到 Gen paper 对齐区间。
若某个 night seed 的 validation split 不稳定，可增加 Codex 生成量再经过 Gen TOF 与
Causal-TOF，例如 `sp_tt` seed2026 使用 100 条 pre-TOF 生成，Gen TOF 后 90 条进入
downstream AD。

Gen 原论文/项目的异常检测参考 F1：

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

小型参考结果文件允许跟随 git：

```text
outputs/reference_gen/anomaly_detection_pipeline_results/
outputs/reference_gen/anomaly_detection_baseline_results/
```

大型 pkl、pth、实验输出和论文 PDF 只作为本地工作区数据，不进入 GitHub。

## GPU 规则

正式实验必须用 GPU。`scripts/run_gen_original_tof.py` 和
`scripts/run_gen_downstream_ad.py` 的结果 JSON 必须记录：

```json
{
  "device": "cuda",
  "requested_device": "cuda"
}
```

如果 Codex 执行环境里出现：

```text
torch.cuda.is_available() is false
RuntimeError: No CUDA GPUs are available
```

这是沙箱权限隔离问题，不是实验应该改成 CPU 的理由。应使用有 GPU 权限的命令执行，
不要把代码改成 CPU fallback，也不要把 CPU 跑出的结果写入主实验。

## Codex 生成规则

本项目的生成模型由 Codex 负责。不要在新文档和新元数据里继续写旧生成口径。正式
生成元数据为：

```json
{
  "generator": "codex_generation",
  "generation_model": "Codex",
  "manual_generation": true
}
```

相关脚本：

```text
scripts/build_codex_generation_package.py
scripts/validate_and_pack_codex_generation.py
```

## 2026-06-26 修复记录

这次排查解决了两个关键问题，后续九格实验不能再犯。

第一，`fr_st` 曾出现 raw causal edge 数为 0，导致 reweighted hints 实际退化成 Gen
transition GSS。根因是 GCAD 风格梯度因果权重 raw scale 很小，旧逻辑先执行
`sparse_threshold`，把正因果边全部清掉。修复点在 `causal_smart_home/causal_prior.py`：
先对正向 causal matrix 做 max-normalize，再执行 sparse threshold。

后续检查 causal GSS 时，`summary.input_causal_edges` 不应为 0。若为 0，优先检查
GCAD 权重归一化、sparse threshold 和输入数据，而不是接受 transition-only GSS。

第二，修出非零 causal edges 后，部分 `sp_st` seed 的 Causal-TOF 被错误惩罚拖低。
根因是 target-distribution guard 已经把过用端点的边标成 `guard_action=downweight`，
旧 Causal-TOF 后处理却仍把这些 downweighted 边当作 violation 惩罚项。修复后的规则：

- downweighted causal edges 仍保留在 hints、scores 和审计字段里。
- 默认不把 `guard_action=downweight` 的边计入 `causal_violation` 惩罚。
- `observed_causal_violation_all_guarded_edges` 用于记录所有 guarded 边上的观测 violation。
- 只有诊断实验才使用 `scripts/run_causal_tof.py --penalize-downweighted-edges`。

## Gen 和 GCAD 的项目化边界

Gen 提供原始 smart-home 数据、two-stage TOF、downstream AD 设置和原论文异常检测参考
分数。GCAD 提供因果关系建模思路。本项目把两者收束成 CausalSmartHome 的单一主流程：

- `causal_smart_home/causal_relation_adapter.py`
- `causal_smart_home/causal_relation_prior_source.py`
- `causal_smart_home/causal_prior.py`
- `causal_smart_home/event_tensor.py`
- `causal_smart_home/causal_gss_reweight.py`
- `causal_smart_home/causal_tof.py`
- `causal_smart_home/gen_core/`
- `causal_smart_home/resources/gen_data/`

主实验脚本默认读取项目内路径。本地完整数据检查：

```bash
python scripts/check_gen_main_data.py
```

当前本地检查结果应为：

```text
GEN_MAIN_DATA_STATUS: ok
cells: 9 (fr, sp, us x spring, night, multiple)
```

## 后续九格执行清单

每个 cell 都按同一口径做：

1. 检查本地 Gen 数据：`python scripts/check_gen_main_data.py`。
2. 构建 causal relation prior。
3. 运行 `scripts/build_causal_gss_prompt.py`，默认添加 causal edges，默认 `guard-mode=downweight`。
4. 用 `scripts/build_codex_generation_package.py` 生成 Codex 输入包。
5. Codex 生成 JSONL 后，用 `scripts/validate_and_pack_codex_generation.py` 打包 pkl。
6. 用 GPU 跑 `scripts/run_gen_original_tof.py`。
7. 跑 `scripts/run_causal_tof.py`。
8. 用 GPU 分别跑 `scripts/run_gen_downstream_ad.py` 的 proposed 和 ablation。
9. 用 `scripts/summarize_main_experiment.py` 输出 per-seed summary。
10. 在 README/实验记录中逐 seed 列出结果，并把 Gen paper AD F1 并排列出。

每个 cell 完成后必须保存：

- 三个 seed 的 proposed 与 ablation 指标。
- Gen paper/project AD F1。
- 命令、config、input manifest。
- `device/requested_device` 字段。
- causal GSS summary，尤其是 `input_causal_edges`。
- Causal-TOF config，尤其是 `penalize_downweighted_edges=false`。

## 清理原则

当前主实验已经确定，旧回滚包、均值/delta、旧生成模型叙事都不再作为主线保留。仓库
只保留主流程脚本、必要测试、README/NOTICE、少量参考结果表和源码。
大型数据、模型、实验输出、临时日志和打包备份不提交到 GitHub。
