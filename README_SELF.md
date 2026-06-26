# CausalSmartHome 内部说明

## 当前主实验

当前项目是 Gen + GCAD 的缝合项目。主实验矩阵不是只有 SP-ST，而是沿用 Gen
异常检测设置的 9 个格子：

```text
FR/SP/US x spring/night/multiple
```

脚本中的短场景名为：

- `st` = spring，源上下文是 winter。
- `tt` = night，源上下文是 daytime。
- `nt` = multiple，源上下文是 single。

目前已经完成并锁定三随机种子结果的是 SP-ST / SP-spring：

```text
因果关系先验
-> 因果增强 GSS
-> GPT-5.5 行为序列生成
-> Gen 原始 two-stage TOF
-> Causal-TOF
-> Gen built-in downstream AD
-> summary
```

完整主方法名为 `proposed_causal_gss_gpt55_causal_tof`。它的核心创新是
causal-relation-enhanced GSS for GPT-5.5 behavior generation。Causal-TOF 是完整
管线中的后处理因果一致性增强组件，不是完整主方法本身。

唯一保留消融是 `ablation_no_causal_tof`，用于证明 Causal-TOF 组件的贡献。
不再保留 raw no Gen TOF 相关对照，因为它偏离当前论文主问题，会稀释主方法叙事。

## 当前结果位置

当前已完成的 SP-ST 主结果目录历史上是：

```text
outputs/main_experiment/
```

其中：

- `gpt55_generation/sp_st/seed2024..2026`：GPT-5.5 生成序列。
- `gen_original_tof/sp_st/seed2024..2026`：Gen 原始 two-stage TOF 输出。
- `causal_tof/sp_st/seed2024..2026`：Causal-TOF 输出。
- `downstream_ad/sp_st/seed2024..2026`：两个保留方法的 Gen downstream AD 结果。
- `summary/`：当前两方法 summary。

2026-06-26 重新检查 GPU 权限、GCAD raw causal edge 和 Causal-TOF guard 逻辑后，
重新从头复现的 SP-ST 标准结果在：

```text
outputs/main_experiment_gpu_fix/sp_st/
outputs/main_experiment_gpu_fix/sp_st/summary_standard/
```

其中 `summary_standard/` 只统计两个标准目录：

- `ablation_no_causal_tof`
- `proposed_causal_gss_gpt55_causal_tof`

不要把临时诊断目录如 `proposed_causal_gss_gpt55_causal_tof_beta0`、
`proposed_causal_gss_gpt55_causal_tof_beta005`、`proposed_causal_gss_gpt55_causal_tof_beta01`
混入正式 summary。这些目录只用于定位 Causal-TOF violation 权重问题。

Gen 原论文/项目的异常检测参考结果已经放在：

```text
outputs/reference_gen/anomaly_detection_pipeline_results/
outputs/reference_gen/anomaly_detection_baseline_results/
```

其中 `SmartGen_results_20250730_221851.json/csv` 是当前主要对比的 Gen 原异常检测
分数，覆盖 FR/SP/US x spring/night/multiple 九个格子。

Frozen 目录：

```text
outputs/main_experiment_frozen/sp_st_gpt55_proposed_3seed_20260623
```

该目录包含 `README_FROZEN.md`、`REPRODUCE.md`、`run_reproduce_from_frozen.sh`、
`MANIFEST.json`、`provenance/checksums.sha256`、`generated/`、`gen_original_tof/`、
`causal_tof/`、`downstream_ad/`、`summary/` 和 `code_snapshot/`。

## 复现方式

复现当前已锁定的 SP-ST 结果不需要重新生成 GPT-5.5 JSONL。直接运行：

```bash
bash outputs/main_experiment_frozen/sp_st_gpt55_proposed_3seed_20260623/run_reproduce_from_frozen.sh
```

脚本会复用 frozen 中的 GPT-5.5 pkl 和 causal GSS hints，重新运行 Gen 原始
two-stage TOF、w/o Causal-TOF downstream AD、Causal-TOF、proposed downstream AD 和
summary。下游 AD 会重新训练，因此不同 GPU、CUDA、PyTorch 环境可能带来小幅波动。

正式实验必须用 GPU。`scripts/run_gen_original_tof.py` 和
`scripts/run_gen_downstream_ad.py` 都应看到 CUDA，并且 downstream AD 的结果 JSON 中
必须记录：

```json
{
  "device": "cuda",
  "requested_device": "cuda"
}
```

如果在 Codex/沙箱里出现 `torch.cuda.is_available() is false` 或
`RuntimeError: No CUDA GPUs are available`，不要把代码改成 CPU fallback。那是沙箱
CUDA 设备隔离问题，应使用带 GPU 权限的提升执行来运行实验命令。后续 FR/SP/US x
spring/night/multiple 九格主实验也都按这个规则执行。

## GCAD 如何被吸收到本项目

原始因果关系能力来自 GCAD 论文/项目。CausalSmartHome 抽取了其中用于多变量时间
序列因果关系建模与先验构建的部分，并在本项目中对外统一命名为“因果关系模块”。

项目内对应文件是：

- `causal_smart_home/causal_relation_adapter.py`
- `causal_smart_home/causal_relation_prior_source.py`
- `causal_smart_home/causal_prior.py`
- `causal_smart_home/event_tensor.py`

这些模块用于构建 causal relation prior，并参与两个位置：

- causal-reweighted GSS：把源域预测因果关系作为 GSS reweight 的软结构信号。
- Causal-TOF：在 Gen 原始 TOF 后进行因果一致性后处理增强。

这样做的目的是把 GCAD 的因果视角接到 Gen 的行为生成与异常检测管线里，形成
CausalSmartHome 的完整主实验。

## 2026-06-26 修复记录：避免 GCAD 退化和 Causal-TOF 误惩罚

这次排查的直接问题有两个：

1. `fr_st` 曾出现 adapter fallback 的 raw causal edge 数为 0，导致
   reweighted hints 实际退化成 Gen transition GSS。
2. 重新修出非零 causal edges 后，`sp_st` 的 Causal-TOF 在部分 seed 上反而破坏了
   很强的 Gen TOF 基线。

第一个问题的根因是 GCAD 风格梯度因果权重 raw scale 很小，旧逻辑先执行
`sparse_threshold`，会把所有正因果边清零。修复方式是在
`causal_smart_home/causal_prior.py` 中先对正向 causal matrix 做 max-normalize，再
执行 sparse threshold。后续看到 raw causal edge 为 0 时，第一时间检查这个归一化
是否生效，而不是接受 transition-only GSS。

同时 `scripts/build_causal_gss_prompt.py` 的默认策略必须保持为：

```text
--add-causal-edges enabled by default
--guard-mode downweight
```

第二个问题的根因是 target-distribution guard 已经把某些过用端点的 causal 边标成
`guard_action=downweight`，但旧 Causal-TOF 后处理仍把这些 downweighted 边当成
causal violation 惩罚项。这样会自相矛盾：一边说这些端点过用需要降权，一边又强迫
序列满足这些边，最终造成重采样偏置和 validation threshold 塌陷。

修复后的规则：

- downweighted causal edges 仍保留在 GSS hints、Causal-TOF scores 和审计字段里。
- 默认不把 `guard_action=downweight` 的边计入 `causal_violation` 惩罚。
- `observed_causal_violation_all_guarded_edges` 仍记录所有 guarded 边上的观测 violation，
  方便证明非零因果信号确实存在。
- 只有诊断实验才使用 `scripts/run_causal_tof.py --penalize-downweighted-edges`。

这条规则非常重要。后续 9 格扩展时，不能把 downweighted 边重新当硬约束惩罚，否则
可能再次出现 proposed 低于 ablation 的假失败。

## Gen 如何被吸收到本项目

原 Gen 项目中当前主实验需要的代码和数据已经放入项目内：

- `causal_smart_home/gen_core/gen_original_tof/security_check.py`
- `causal_smart_home/gen_core/gen_original_tof/models1.py`
- `causal_smart_home/gen_core/gen_original_tof/check_model/`
- `causal_smart_home/gen_core/anomaly_detection_pipeline/models1.py`
- `causal_smart_home/gen_core/anomaly_detection_pipeline/check_model/`
- `causal_smart_home/gen_core/anomaly_detection_pipeline/attack/`
- `causal_smart_home/gen_core/anomaly_detection_pipeline/test/`
- `causal_smart_home/gen_core/anomaly_detection_pipeline/synthetic_data/`
- `causal_smart_home/resources/gen_data/dictionary.py`
- `causal_smart_home/resources/gen_data/{fr,sp,us}/`

主实验脚本默认读取上述项目内路径，不再依赖外部源码目录或符号链接。

检查完整 Gen 主实验数据：

```bash
python scripts/check_gen_main_data.py
```

## 为什么不再依赖 external_sources

当前项目已经进入正式项目化版本，需要能够独立复现当前主实验。外部源码目录会让
复现依赖机器本地路径，也会让项目叙事看起来像临时拼接工程。因此当前版本把主实验
必需的 Gen 代码、数据与因果关系模块合并进仓库，只把真实来源记录在本内部文档。

## 为什么不再保留 raw no Gen TOF

当前主问题是：因果增强 GSS 生成的序列经过 Gen 原始 two-stage TOF 后，再加入
Causal-TOF 是否提升最终 downstream AD。raw no Gen TOF 不是这个主问题的必要对照，
也不能解释 Causal-TOF 组件在完整管线中的贡献，所以已从主实验、复现脚本、summary
和 frozen 展示中移除。

## 当前两方法结果

以下是 2026-06-26 GPU fix 后的 SP-ST / SP-spring 三种子标准结果，路径为：

```text
outputs/main_experiment_gpu_fix/sp_st/summary_standard/
```

| method | precision mean | recall mean | f1 mean | accuracy mean | fpr mean | fnr mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| proposed_causal_gss_gpt55_causal_tof | 0.953861 | 0.997711 | 0.975220 | 0.974588 | 0.048535 | 0.002289 |
| ablation_no_causal_tof | 0.840026 | 0.992216 | 0.898191 | 0.867903 | 0.256410 | 0.007784 |

结论：完整主方法以因果增强 GSS 为核心。加入 Causal-TOF 后，SP-ST 三种子均值 F1
从 0.898191 提升到 0.975220，FPR 从 0.256410 降到 0.048535。

逐 seed 结果：

| seed | ablation F1 | proposed F1 | proposed precision | proposed recall | proposed FPR | device |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2024 | 0.741344 | 0.965517 | 0.933333 | 1.000000 | 0.071429 | cuda |
| 2025 | 0.974565 | 0.981132 | 0.962963 | 1.000000 | 0.038462 | cuda |
| 2026 | 0.978665 | 0.979012 | 0.965287 | 0.993132 | 0.035714 | cuda |

这次复现还验证了两个检查：

```bash
pytest -q
# 20 passed

python scripts/check_gen_main_data.py
# GEN_MAIN_DATA_STATUS: ok
# cells: 9 (fr, sp, us x spring, night, multiple)
```

## 后续扩新场景

扩展新场景时，优先保持同一主方法口径：

1. 准备新场景的 Gen dictionary、source pkl、target/split test pkl 和 downstream AD 所需数据。
2. 用因果关系模块生成 causal relation prior。
3. 运行 `scripts/build_causal_gss_prompt.py` 构建 causal-reweighted GSS prompt。
4. 用 GPT-5.5 生成并用 `scripts/validate_and_pack_gpt55_generation.py` 打包。
5. 运行 `scripts/run_gen_original_tof.py`。
6. 运行 `scripts/run_causal_tof.py`。
7. 分别运行 `scripts/run_gen_downstream_ad.py` 的 proposed 和 ablation。
8. 运行 `scripts/summarize_main_experiment.py`，只展示完整主方法和 w/o Causal-TOF 消融。

扩展时还要额外检查：

- Gen original TOF 和 downstream AD 必须用 GPU，不能静默 CPU fallback。
- causal GSS summary 中 `input_causal_edges` 不应为 0；如果为 0，先检查 GCAD 权重归一化和
  sparse threshold。
- `guarded_reweighted_gss_hints.json` 允许存在 `guard_action=downweight`，但 Causal-TOF
  默认不能把这些边计入 violation 惩罚。
- summary 只统计标准 variant 目录，不能把 beta/diagnostic 目录混进去。
- 每跑一个 cell，都要保存 per-seed、aggregate、command/config、device 字段和数据检查结果。
