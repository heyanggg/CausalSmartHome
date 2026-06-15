# CausalSmartHome

> 非侵入式“胶水层”项目：把 **SmartGen**、**SmartGuard** 与 **GCAD-style Granger causality** 串起来，用因果先验约束智能家居行为生成与过滤，再接入原异常检测流程。

## 1. 项目定位

`CausalSmartHome` 不是一个替代 SmartGen、SmartGuard 或 GCAD 的新大模型，也不是把三份源码强行揉在一起的魔改项目。它的定位是：

* **SmartGen**：负责从原上下文行为中生成目标上下文正常行为数据；
* **GCAD-style causal prior**：负责从正常行为中挖掘 action/device 级 Granger-style 因果先验；
* **CausalSmartHome**：负责事件张量桥接、因果先验构建、因果一致性过滤和实验脚本粘合；
* **SmartGuard / SmartGen anomaly detection pipeline**：负责下游异常检测训练与评估。

核心原则：

```text
不魔改原项目主体代码
只在输入输出边界做 wrapper / adapter / filter
失败项目 CausalGenGuard 不纳入当前方案
```

## 2. 背景：三篇论文如何被“缝合”

### 2.1 SmartGen

SmartGen 面向智能家居行为漂移场景，用 LLM 合成目标上下文行为数据。它的主流程包括：

* **TSS**：Time and Semantic-aware Split，把长行为序列切成更适合 LLM 的短序列；
* **SSC / SPPC**：Semantic-aware Sequence Compression，压缩冗余行为序列；
* **GSS**：Graph-guided Sequence Synthesis，用行为转移图和 JSON hints 引导 LLM 生成；
* **TOF**：Two-stage Outlier Filter，过滤不合理或低价值的合成序列。

SmartGen 适合解决季节、作息、人数变化导致的目标上下文训练数据不足问题，但它的 GSS 主要是转移频率约束，不显式表达滞后、方向性和非对称的因果结构。

### 2.2 SmartGuard

SmartGuard 面向智能家居用户行为异常检测。它使用 Transformer Autoencoder 重构正常行为序列，并通过重构损失判断异常。核心模块包括：

* **LDMS**：Loss-guided Dynamic Mask Strategy，鼓励模型学习低频但正常的 hard-to-learn 行为；
* **TTPE**：Three-level Time-aware Position Embedding，建模顺序、时刻、持续时间；
* **NWRL**：Noise-aware Weighted Reconstruction Loss，降低噪声行为对异常分数的干扰。

SmartGuard 对 SD、MD、DM、DD 四类异常有效，但如果只用原上下文正常数据训练，遇到目标上下文正常行为漂移时可能出现误报。

### 2.3 GCAD

GCAD 面向多变量时间序列异常检测。它训练预测器，然后对每个输出通道的预测误差分别反向传播，利用输入窗口上的梯度衡量变量间 Granger-style 预测影响，并通过非对称稀疏化得到因果图。

GCAD 的思想适合补足 SmartGen 的结构约束，但原 GCAD 输入是连续多变量时间序列，而智能家居行为是离散事件序列。因此本项目引入 **Event Tensor Bridge**，把行为序列映射到 `[T, C]` 多变量事件张量。

## 3. 方法总览

整体流程：

```text
SmartGen / SmartGuard 行为序列
        |
        v
[day, hour_slot, device_id, action_id] 四元组解析
        |
        v
Event Tensor Bridge
        |
        v
GCAD-style gradient causal prior
        |
        +------------------------------+
        |                              |
        v                              v
causal prompt hints              causal consistency filter
        |                              |
        v                              v
SmartGen generation / TOF       filtered synthetic sequences
        |                              |
        +---------------> SmartGen / SmartGuard anomaly detection pipeline
```

当前实现中，因果先验主要用于 **生成后过滤**。生成前 causal prompt 的接口已经保留，但当前 FR spring 实验主要验证的是 filter 路径。

## 4. 当前代码结构

当前仓库采用 root package 布局：

```text
CausalSmartHome/
  causal_smart_home/
    __init__.py
    schema.py
    event_tensor.py
    causal_prior.py
    gcad_adapter.py
    smartgen_adapter.py
    smartguard_adapter.py
    causal_prompt.py
    causal_filter.py
    pipeline.py
    demo_data.py
    cli.py
  docs/
    task1_source_review.md
    task2_feasibility_and_plan.md
    task3_chinese_paper_draft.md
    task4_implementation_and_testing.md
    figures/framework.svg
  external_sources/
    SmartGuard -> /home/heyang/projects/SmartGuard
    SmartGen   -> /home/heyang/projects/SmartGen
    GCAD       -> /home/heyang/projects/GCAD
  outputs/
  tests/
  README.md
  pyproject.toml
```

主要模块说明：

| 模块                      | 作用                                 |
| ----------------------- | ---------------------------------- |
| `schema.py`             | 统一行为四元组和行为序列表示                     |
| `event_tensor.py`       | 将离散行为序列转为 `[T, C]` 事件张量            |
| `causal_prior.py`       | 轻量 GCAD-style 梯度因果先验挖掘与序列化         |
| `gcad_adapter.py`       | GCAD-style 因果挖掘包装                  |
| `smartgen_adapter.py`   | 复用 SmartGen pkl 数据约定               |
| `smartguard_adapter.py` | 以 subprocess 方式包装原 SmartGuard/下游脚本 |
| `causal_prompt.py`      | 将因果先验转为 SmartGen prompt hints      |
| `causal_filter.py`      | 对 SmartGen 生成序列做因果一致性评分和过滤         |
| `pipeline.py`           | 端到端流程编排                            |
| `cli.py`                | 命令行入口                              |

## 5. 环境说明

当前服务器环境：

```text
项目路径：/home/heyang/projects/CausalSmartHome
Conda 环境：smartguard_env
Python：3.8.20
```

由于当前 `pyproject.toml` 要求 Python >= 3.9，而服务器环境是 Python 3.8，因此不要优先使用 `pip install -e .`。推荐在项目根目录用：

```bash
cd /home/heyang/projects/CausalSmartHome
conda activate smartguard_env
PYTHONPATH=. python -m causal_smart_home.cli --help
```

## 6. external_sources 设置

推荐把 `external_sources` 指向同级目录中完整 clone / 修复后的原项目，而不是使用旧 tar 包快照：

```bash
cd /home/heyang/projects/CausalSmartHome

mv external_sources external_sources_snapshot_from_tar
mkdir external_sources

ln -s /home/heyang/projects/SmartGuard external_sources/SmartGuard
ln -s /home/heyang/projects/SmartGen external_sources/SmartGen
ln -s /home/heyang/projects/GCAD external_sources/GCAD

readlink -f external_sources/SmartGuard
readlink -f external_sources/SmartGen
readlink -f external_sources/GCAD
```

期望输出：

```text
/home/heyang/projects/SmartGuard
/home/heyang/projects/SmartGen
/home/heyang/projects/GCAD
```

注意：`build-prior`、`prompt`、`filter` 本身主要吃 `.pkl` 和 `.json` 路径，不强依赖 `external_sources`。但后续调用原 SmartGen/SmartGuard pipeline 时，软链接能让路径更稳定。

## 7. CLI 用法

### 7.1 查看帮助

```bash
cd /home/heyang/projects/CausalSmartHome
conda activate smartguard_env

PYTHONPATH=. python -m causal_smart_home.cli --help
```

### 7.2 构建因果先验

```bash
PYTHONPATH=. python -m causal_smart_home.cli build-prior \
  --train-pkl /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/winter/split_trn.pkl \
  --out-dir outputs/fr_winter_to_spring_device_h5e-05 \
  --lag 4 \
  --epochs 40 \
  --level device \
  --sparse-threshold 0.00005
```

当前更推荐的做法是：先用 `--sparse-threshold 0.0` 训练得到完整矩阵，再离线派生不同阈值版本，避免重复训练。

### 7.3 对 SmartGen 生成序列做因果过滤

```bash
PYTHONPATH=. python -m causal_smart_home.cli filter \
  --prior-json outputs/fr_winter_to_spring_device_h5e-05/causal_prior.json \
  --generated-pkl /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/spring/fr_spring_generation_SPPC_th=0.918_gpt-4o_seq.pkl \
  --out-pkl outputs/fr_winter_to_spring_device_h5e-05/fr_spring_generation_causal_kept.pkl \
  --out-scores outputs/fr_winter_to_spring_device_h5e-05/fr_spring_generation_causal_scores.json \
  --min-coverage 0.5
```

### 7.4 对 SmartGen 原 `filter_true` 再做因果过滤

这是当前更公平的下游对比方式：

```bash
PYTHONPATH=. python -m causal_smart_home.cli filter \
  --prior-json outputs/fr_winter_to_spring_device_h5e-05/causal_prior.json \
  --generated-pkl /home/heyang/projects/SmartGen/anomaly_detection_pipeline/synthetic_data/fr_spring_generation_SPPC_th=0.918_gpt-4o_seq_filter_true.pkl \
  --out-pkl outputs/fr_winter_to_spring_device_h5e-05/fr_spring_SPPC_filter_true_causal_kept.pkl \
  --out-scores outputs/fr_winter_to_spring_device_h5e-05/fr_spring_SPPC_filter_true_causal_scores.json \
  --min-coverage 0.5
```

## 8. 已完成实验总结

### 8.1 action-level 因果先验

最初使用：

```text
level = action
sparse-threshold = 0.001
```

结果因果矩阵全零。原因是边权最大只有约 `1e-4`，阈值 `0.001` 太高。

改成：

```text
level = action
sparse-threshold = 0.0
```

结果：

```text
channels = 40
nonzero offdiag = 780
max offdiag ≈ 0.0001095

raw = 137
kept = 128
rejected = 9
reject_ratio = 6.57%
checked_nonzero = 11 / 137 = 8.03%
```

结论：action-level 过滤有效，但短序列下命中率偏低。

### 8.2 device-level 因果先验

使用：

```text
level = device
lag = 4
epochs = 40
sparse-threshold = 0.0
```

结果：

```text
channels = 15
nonzero offdiag = 105
max offdiag = 0.0005482730339281261

raw = 137
kept = 79
rejected = 58
reject_ratio = 42.34%
checked_nonzero = 92 / 137 = 67.15%
checked_total = 233
```

结论：device-level 比 action-level 更容易命中短序列中的因果边，但 h0 保留太多弱边，过滤过严。

### 8.3 device 阈值扫描

从 device_h0 派生多个阈值，不重新训练：

| setting       | nonzero offdiag | kept | rejected | reject_ratio | checked_nonzero_ratio | checked_total |
| ------------- | --------------: | ---: | -------: | -----------: | --------------------: | ------------: |
| device_h0     |             105 |   79 |       58 |       42.34% |                67.15% |           233 |
| device_h1e-05 |              24 |   79 |       58 |       42.34% |                66.42% |           228 |
| device_h2e-05 |              17 |   85 |       52 |       37.96% |                62.04% |           140 |
| device_h5e-05 |              11 |  119 |       18 |       13.14% |                13.87% |            32 |
| device_h1e-04 |               8 |  119 |       18 |       13.14% |                13.87% |            31 |

当前主设置：

```text
level = device
lag = 4
epochs = 40
sparse_threshold = 5e-05
```

理由：`device_h5e-05` 不是空过滤，也不过度过滤；相比 `h1e-04` 保留更多因果边，但过滤效果相同。

## 9. 下游异常检测结果

### 9.1 不公平对比：CausalGCAD only

直接用 raw SmartGen generated sequence 做因果过滤，然后接 anomaly detection：

| Method          | Recall | Precision |     F1 |
| --------------- | -----: | --------: | -----: |
| Original SPPC   | 0.9886 |    0.7311 | 0.8406 |
| CausalGCAD only | 0.3409 |    0.4839 | 0.4000 |

结论：这不是合理融合方式。原因是 Original SPPC 使用 SmartGen 已经过 TOF / filter_true 的高质量数据，而 CausalGCAD only 用的是 raw generated sequence + 因果过滤，数据基础不公平。

### 9.2 合理对比：SPPC filter_true + CausalGCAD

正确流程：

```text
SmartGen 原生成
  -> SmartGen 原 filter_true
  -> CausalGCAD 因果过滤
  -> anomaly detection pipeline
```

对 `fr_spring_generation_SPPC_th=0.918_gpt-4o_seq_filter_true.pkl` 过滤结果：

```text
original_filter_true = 125
causal_kept = 109
rejected = 16
reject_ratio = 12.8%
checked_nonzero = 17 / 125 = 13.6%
checked_total = 30
```

下游结果：

| Method                | Recall | Precision |     F1 |
| --------------------- | -----: | --------: | -----: |
| Original SPPC         | 0.9886 |    0.7311 | 0.8406 |
| SPPC_CausalGCAD hard  | 1.0000 |    0.6822 | 0.8111 |
| SPPC_CausalGCAD Cons2 | 1.0000 |    0.6822 | 0.8111 |
| SPPC_CausalGCAD Cons3 | 1.0000 |    0.6822 | 0.8111 |

其中：

```text
Cons2: 只删除 causal_coverage < 0.5 且 num_checked_edges >= 2 的样本
Cons3: 只删除 causal_coverage < 0.5 且 num_checked_edges >= 3 的样本
```

结论：

```text
当前 FR winter -> spring 单组实验没有证明 F1 提升。
因果过滤提升了 Recall，但降低了 Precision。
更准确地说：因果过滤让模型对异常更敏感，但引入了更多误报。
```

关键现象：

```text
Original SPPC validation avg loss ≈ 0.1578
Original SPPC threshold ≈ 1.3921

SPPC_CausalGCAD validation avg loss ≈ 0.0040
SPPC_CausalGCAD threshold ≈ 0.0121
```

这说明因果过滤后的训练/验证数据过于“干净”或分布变窄，导致 anomaly threshold 过低，测试时更多正常样本被判为异常，从而 precision 下降。

## 10. 当前阶段结论

1. 工程链路已经跑通：SmartGen 数据 -> GCAD-style causal prior -> 因果过滤 -> SmartGen anomaly detection pipeline。
2. `device_h5e-05` 是目前最合理的主因果先验设置。
3. 直接用 CausalGCAD 替代 SmartGen filter_true 不合理，效果明显变差。
4. 更合理的融合方式是 `SPPC filter_true + GCAD causal filtering`。
5. 当前 FR spring 单组下，因果过滤没有超过原 SPPC 的 F1。
6. 当前最诚实的结果表述是：因果过滤提高召回率，但降低精确率；硬删除式过滤需要校准或改成软过滤。
7. 下一阶段不应直接宣称“性能提升”，而应聚焦在阈值校准、软权重过滤、多数据集验证和误报机制解释。

## 11. 推荐后续工作

暂时不继续的工作：

```text
先不扫 anomaly detection percentile t
```

后续可以做：

1. **阈值校准**：扫描 anomaly detection 的 percentile `t`，因为因果过滤后 validation loss 和 threshold 明显变低。
2. **软过滤**：不直接删除样本，而是把 `causal_coverage` 作为训练权重或样本排序指标。
3. **双目标选择**：同时考虑 SmartGen TOF score 和 causal coverage，而不是只用 hard causal rule。
4. **扩展数据集**：跑 `sp/night/us/multiple`，确认因果过滤是否只在 FR spring 单点无提升。
5. **解释性分析**：展示被删样本违反了哪些 causal edges，以及这些边是否对应异常检测中的高 loss 行为。
6. **写论文时调整叙事**：从“性能提升”改为“因果约束可解释地改变召回-精确率权衡，并暴露合成数据过滤与阈值校准之间的关系”。

## 12. 常用命令备忘

### 12.1 build prior

```bash
cd /home/heyang/projects/CausalSmartHome
conda activate smartguard_env

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
PYTHONPATH=. python -m causal_smart_home.cli build-prior \
  --train-pkl /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/winter/split_trn.pkl \
  --out-dir outputs/fr_winter_to_spring_device_h0 \
  --lag 4 \
  --epochs 40 \
  --level device \
  --sparse-threshold 0.0
```

### 12.2 filter raw SmartGen generation

```bash
PYTHONPATH=. python -m causal_smart_home.cli filter \
  --prior-json outputs/fr_winter_to_spring_device_h5e-05/causal_prior.json \
  --generated-pkl /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/spring/fr_spring_generation_SPPC_th=0.918_gpt-4o_seq.pkl \
  --out-pkl outputs/fr_winter_to_spring_device_h5e-05/fr_spring_generation_causal_kept.pkl \
  --out-scores outputs/fr_winter_to_spring_device_h5e-05/fr_spring_generation_causal_scores.json \
  --min-coverage 0.5
```

### 12.3 filter SmartGen filter_true

```bash
PYTHONPATH=. python -m causal_smart_home.cli filter \
  --prior-json outputs/fr_winter_to_spring_device_h5e-05/causal_prior.json \
  --generated-pkl /home/heyang/projects/SmartGen/anomaly_detection_pipeline/synthetic_data/fr_spring_generation_SPPC_th=0.918_gpt-4o_seq_filter_true.pkl \
  --out-pkl outputs/fr_winter_to_spring_device_h5e-05/fr_spring_SPPC_filter_true_causal_kept.pkl \
  --out-scores outputs/fr_winter_to_spring_device_h5e-05/fr_spring_SPPC_filter_true_causal_scores.json \
  --min-coverage 0.5
```

### 12.4 copy into SmartGen anomaly detection pipeline

```bash
cd /home/heyang/projects

cp \
  /home/heyang/projects/CausalSmartHome/outputs/fr_winter_to_spring_device_h5e-05/fr_spring_SPPC_filter_true_causal_kept.pkl \
  /home/heyang/projects/SmartGen/anomaly_detection_pipeline/synthetic_data/fr_spring_generation_SPPC_CausalGCAD_th=5e-05_gpt-4o_seq_filter_true.pkl
```

### 12.5 run downstream anomaly detection

```bash
cd /home/heyang/projects/SmartGen/anomaly_detection_pipeline
conda activate smartguard_env

python - <<'PY'
import json
from pathlib import Path
from Anomaly_Detection_pipeline_model import Anomaly_detection

result = Anomaly_detection(
    dataset="fr",
    new_env="spring",
    thres="5e-05",
    method="SPPC_CausalGCAD",
    model="gpt-4o",
    t=95.5,
)

print(json.dumps(result, indent=2))

out = Path("/home/heyang/projects/CausalSmartHome/outputs/fr_winter_to_spring_device_h5e-05/anomaly_detection_SPPC_CausalGCAD_fr_spring.json")
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
PY
```

## 13. 论文写作建议

可以写：

```text
我们实现了一个非侵入式因果胶水层，将离散智能家居行为映射为事件张量，并从原上下文正常行为中抽取 device/action 级 Granger-style causal prior。该先验可用于过滤 SmartGen 合成数据。FR winter -> spring 的实验显示，因果过滤会显著改变下游异常检测的召回-精确率权衡：召回率提升到 1.0，但精确率下降，最终 F1 未超过原 SPPC。这说明简单 hard deletion 会使训练/验证分布过窄，需要进一步进行阈值校准或软权重过滤。
```

不要写：

```text
因果过滤已经提升了 SmartGuard/SmartGen 的异常检测性能。
```

当前数据不支持这个结论。

## 14. 当前主结果文件

```text
outputs/fr_winter_to_spring_device_h5e-05/causal_prior.json
outputs/fr_winter_to_spring_device_h5e-05/fr_spring_generation_causal_kept.pkl
outputs/fr_winter_to_spring_device_h5e-05/fr_spring_generation_causal_scores.json
outputs/fr_winter_to_spring_device_h5e-05/fr_spring_SPPC_filter_true_causal_kept.pkl
outputs/fr_winter_to_spring_device_h5e-05/fr_spring_SPPC_filter_true_causal_scores.json
outputs/fr_winter_to_spring_device_h5e-05/anomaly_detection_SPPC_CausalGCAD_fr_spring.json
outputs/fr_winter_to_spring_device_h5e-05/anomaly_detection_SPPC_CausalGCADCons2_fr_spring.json
outputs/fr_winter_to_spring_device_h5e-05/anomaly_detection_SPPC_CausalGCADCons3_fr_spring.json
```

## 15. 一句话总结

`CausalSmartHome` 已经完成了三篇论文的工程缝合：SmartGen 提供目标上下文生成数据，GCAD-style 方法提供可解释因果先验，SmartGuard / SmartGen anomaly detection pipeline 负责下游检测。当前 FR spring 实验表明，因果过滤能提高异常召回率，但会降低精确率，F1 尚未超过原 SPPC；下一步重点应从“硬删除”转向“软权重、阈值校准和多场景验证”。
