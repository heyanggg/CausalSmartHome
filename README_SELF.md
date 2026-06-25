# CausalSmartHome 内部说明

## 当前主实验

当前主实验是 SP-ST 三随机种子实验：

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

主结果目录：

```text
outputs/main_experiment/
```

其中：

- `gpt55_generation/sp_st/seed2024..2026`：GPT-5.5 生成序列。
- `gen_original_tof/sp_st/seed2024..2026`：Gen 原始 two-stage TOF 输出。
- `causal_tof/sp_st/seed2024..2026`：Causal-TOF 输出。
- `downstream_ad/sp_st/seed2024..2026`：两个保留方法的 Gen downstream AD 结果。
- `summary/`：当前两方法 summary。

Frozen 目录：

```text
outputs/main_experiment_frozen/sp_st_gpt55_proposed_3seed_20260623
```

该目录包含 `README_FROZEN.md`、`REPRODUCE.md`、`run_reproduce_from_frozen.sh`、
`MANIFEST.json`、`provenance/checksums.sha256`、`generated/`、`gen_original_tof/`、
`causal_tof/`、`downstream_ad/`、`summary/` 和 `code_snapshot/`。

## 复现方式

复现当前主实验不需要重新生成 GPT-5.5 JSONL。直接运行：

```bash
bash outputs/main_experiment_frozen/sp_st_gpt55_proposed_3seed_20260623/run_reproduce_from_frozen.sh
```

脚本会复用 frozen 中的 GPT-5.5 pkl 和 causal GSS hints，重新运行 Gen 原始
two-stage TOF、w/o Causal-TOF downstream AD、Causal-TOF、proposed downstream AD 和
summary。下游 AD 会重新训练，因此不同 GPU、CUDA、PyTorch 环境可能带来小幅波动。

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

## Gen 如何被吸收到本项目

原 Gen 项目中当前主实验需要的代码和数据已经放入项目内：

- `causal_smart_home/gen_core/SmartGen/security_check.py`
- `causal_smart_home/gen_core/SmartGen/check_model/`
- `causal_smart_home/gen_core/anomaly_detection_pipeline/models1.py`
- `causal_smart_home/gen_core/anomaly_detection_pipeline/check_model/`
- `causal_smart_home/gen_core/anomaly_detection_pipeline/attack/`
- `causal_smart_home/gen_core/anomaly_detection_pipeline/test/`
- `causal_smart_home/resources/gen_data/dictionary.py`
- `causal_smart_home/resources/gen_data/sp/winter/trn.pkl`
- `causal_smart_home/resources/gen_data/sp/spring/split_test.pkl`
- `causal_smart_home/resources/gen_data/sp/spring/test.pkl`
- `causal_smart_home/resources/gen_data/sp/spring/trn.pkl`
- `causal_smart_home/resources/gen_data/sp/spring/vld.pkl`

主实验脚本默认读取上述项目内路径，不再依赖外部源码目录或符号链接。

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

| method | precision mean | recall mean | f1 mean | accuracy mean | fpr mean | fnr mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| proposed_causal_gss_gpt55_causal_tof | 0.953861 | 0.997711 | 0.975220 | 0.974588 | 0.048535 | 0.002289 |
| ablation_no_causal_tof | 0.840026 | 0.992216 | 0.898191 | 0.867903 | 0.256410 | 0.007784 |

结论：完整主方法以因果增强 GSS 为核心。加入 Causal-TOF 后，SP-ST 三种子均值 F1
从 0.898191 提升到 0.975220，FPR 从 0.256410 降到 0.048535。

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
