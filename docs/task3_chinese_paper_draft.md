# 任务三：论文初稿（中文版）

# 基于子序列因果约束的上下文自适应智能家居异常检测

## 摘要

智能家居系统中的设备行为直接作用于用户生活空间，异常行为可能来自用户误操作，也可能来自攻击者对设备或平台的恶意控制。现有用户行为异常检测方法通常只用静态历史正常数据训练，当用户行为因季节、作息或家庭人数变化而发生漂移时，新的正常行为容易被误判为异常。因此，本文研究的问题是：如何在不重新大量采集目标上下文真实数据的情况下，提高智能家居异常检测模型对行为漂移的适应能力。

现有方法存在两个不足。SmartGuard 能通过时间感知自编码器检测行为异常，但训练分布固定，对上下文漂移敏感。SmartGen 能利用大语言模型生成目标上下文行为数据，用于下游模型再训练，但其图引导生成主要依赖行为转移频率，难以表达行为子序列之间的非对称、滞后因果关系。GCAD 能从多变量时间序列预测器梯度中发现 Granger 因果图，但它面向连续传感器时间序列，不能直接处理智能家居离散行为序列。

本文提出一种非侵入式胶水框架 CausalSmartHome。框架首先把离散行为序列转换为多变量事件时间序列，使动作、设备或设备-动作对成为变量通道；然后使用 GCAD 风格的梯度 Granger 因果挖掘方法，从原上下文正常行为中得到子序列级因果先验；接着将该因果先验转化为 JSON 形式的 causal hints，与 SmartGen 的行为转移提示共同输入大语言模型；最后在 SmartGen 的两阶段异常过滤之后增加因果一致性过滤，保留更符合历史行为因果结构的合成序列，并用这些序列增广 SmartGuard 的训练数据。

实验将围绕 FR、SP、US 三个智能家居数据集，在季节迁移、作息迁移和人数迁移三类上下文变化下评估本文方法。主要比较 SmartGuard 原始训练、SmartGen 增广训练、SmartGen+TOF 增广训练以及本文因果增强增广训练。实验指标包括 Precision、Recall、F1、FPR、FNR 和生成序列因果覆盖率。预期实验将说明：在保持 SmartGuard 模型不变的前提下，引入子序列因果约束可以减少上下文漂移导致的误报，并提升合成数据对异常检测再训练的有效性。

---

## 1 Introduction

智能家居设备越来越多地参与用户日常生活，例如照明、门锁、摄像头、空调、水阀、窗帘和厨房电器等。与普通网络系统不同，智能家居异常不仅影响信息安全，也可能直接影响物理安全。例如，攻击者关闭摄像头并打开窗户可能导致入侵风险；用户长时间忘记关闭水阀可能导致漏水；冬季误开空调制冷可能造成不适。因此，智能家居异常检测需要理解用户行为序列中的设备组合、操作时刻和持续时间，及时发现偏离正常生活规律的行为。

已有研究已经从不同角度提升了智能家居行为建模能力。SmartGuard 使用 Transformer Autoencoder 重构正常行为序列，通过 Loss-guided Dynamic Mask Strategy 学习低频难学行为，通过 Three-level Time-aware Position Embedding 建模顺序、时刻和持续时间，通过 Noise-aware Weighted Reconstruction Loss 降低噪声行为对推理的干扰。SmartGen 进一步关注行为漂移问题，使用大语言模型生成目标上下文下的正常行为序列，并通过 Time and Semantic-aware Split、Semantic-aware Sequence Compression、Graph-guided Sequence Synthesis 和 Two-stage Outlier Filter 提高生成数据质量。另一方面，GCAD 证明了在多变量时间序列异常检测中，动态 Granger 因果模式偏离可以作为有效异常信号。

然而，现有方法仍然缺少一个关键环节：上下文自适应生成数据不仅要看起来符合新环境，还应保留用户历史行为中的高阶结构关系。SmartGen 的 GSS 可以记录常见行为转移，但转移频率主要描述相邻动作共现，不能充分表达多步滞后和方向性影响。SmartGuard 的 TTPE 能检测时刻和持续时间异常，但它没有显式利用生成数据中的因果结构来约束再训练样本。GCAD 虽然能挖掘因果图，但原始输入是连续多变量传感器时间序列，而智能家居用户行为是离散事件序列。如果直接把行为编号作为连续数值输入 GCAD，会破坏数据语义。

为了解决上述问题，本文提出 CausalSmartHome，一种面向智能家居行为漂移异常检测的非侵入式缝合框架。本文不修改 SmartGuard、SmartGen 和 GCAD 的主体代码，而是在三者输入输出边界建立胶水层。具体而言，本文设计事件张量桥接方法，把行为序列转换为多变量事件时间序列；在该事件张量上挖掘 GCAD 风格的子序列 Granger 因果先验；将因果边以 JSON hints 的形式注入 SmartGen prompt；并在生成后增加因果一致性过滤。经过过滤的目标上下文合成序列用于 SmartGuard 再训练，从而提升其在行为漂移场景下的鲁棒性。

本文贡献如下：

1. 提出一种离散智能家居行为序列到多变量事件时间序列的桥接方法，使 GCAD 风格的梯度 Granger 因果挖掘可以用于行为子序列结构建模。
2. 提出 Causal-GSS 生成约束，将 SmartGen 的转移频率提示与子序列因果先验结合，使 LLM 合成数据同时满足上下文变化和历史行为结构约束。
3. 提出生成后因果一致性过滤机制，在 SmartGen TOF 之后剔除局部合理但违反高权重因果边的序列，提高合成数据对 SmartGuard 再训练的有效性。
4. 设计一个不修改原始三项目模型代码的胶水框架，保留 SmartGuard 的异常检测能力、SmartGen 的上下文生成能力和 GCAD 的可解释因果建模能力。
5. 给出面向 FR、SP、US 数据集和 ST/TT/NT 上下文漂移的实验方案，并设计针对 SD/MD/DM/DD 智能家居异常的系统评估。

---

## 2 Related Work

### 2.1 智能家居用户行为异常检测

智能家居异常检测旨在发现与正常生活规律不一致的设备操作序列。早期方法通常依赖规则、统计模型或 Markov 转移关系，能够捕获部分常见设备操作模式，但难以建模复杂时间上下文和多设备组合关系。随着深度学习方法的发展，GRU、LSTM、Transformer Autoencoder 等模型被用于用户行为建模，通过重构误差或预测误差识别异常。SmartGuard 是该方向的代表方法之一，它针对智能家居行为的低频行为、时间上下文和噪声行为分别设计 LDMS、TTPE 与 NWRL，从而提升无监督异常检测效果。

这些方法的主要局限是训练分布通常固定。当季节、作息或入住人数变化后，用户行为的设备分布、操作时刻和行为密度都会改变。若模型仍只使用原上下文正常数据训练，则可能把目标上下文下的正常行为误判为异常。因此，异常检测模型需要适应行为漂移。

### 2.2 智能家居行为数据合成与上下文自适应

为缓解目标上下文数据采集成本高、周期长和隐私敏感的问题，SmartGen 使用 LLM 合成目标上下文用户行为序列。它先通过 TSS 切分长序列，避免行为模式混杂；再通过 SSC/SPPC 压缩语义相似序列，降低 LLM 输入长度；同时通过 GSS 构建行为转移图，保留用户习惯频率信息；最后用 TOF 剔除不合理生成序列。该流程能够为下游异常检测和行为预测模型提供增广训练数据。

不过，生成数据的合理性不只取决于单步转移频率。许多智能家居行为存在多步依赖和方向性，例如到家后解锁、开灯、做饭、吃饭、洗碗这类行为链。若生成序列仅满足局部相邻转移，但颠倒关键行为链顺序，仍可能降低下游异常检测模型质量。因此，需要在 SmartGen 生成过程中加入更高阶的结构约束。

### 2.3 多变量时间序列因果异常检测

多变量时间序列异常检测常关注变量间依赖结构。GNN 方法通常通过自适应图学习变量相关性，再用预测或重构误差识别异常。然而，相关性图往往缺少方向性和解释性。GCAD 从 Granger 因果角度出发，用深度预测器的通道分离梯度衡量变量间预测影响，并通过对称稀疏化保留非对称方向边。测试时，GCAD 通过当前因果图与正常因果图之间的偏离计算异常分数。

GCAD 的思想适合补足智能家居生成数据中的结构约束，但不能直接应用于离散行为 id。本文通过事件张量桥接将行为转化为多变量事件时间序列，从而把 GCAD 的因果挖掘能力迁移到智能家居行为子序列层面。

---

## 3 方法框架总览

本文提出的 CausalSmartHome 是一个非侵入式框架。它不改变 SmartGuard 的 LDMS/TTPE/NWRL，不改变 SmartGen 的 TSS/SSC/GSS/TOF，也不改变 GCAD 的梯度因果思想，而是在三者之间增加数据和提示词胶水。

框架图如下：

![CausalSmartHome Framework](figures/framework.svg)

整体流程如下。

首先，输入原上下文正常行为序列。SmartGen 按原流程执行 TSS 切分、SSC/SPPC 压缩和 GSS 转移图统计，得到压缩后的代表性行为序列和高频转移提示。同时，胶水层把同一批正常行为序列转换为事件张量，在该张量上训练预测器并计算 GCAD 风格 Granger 因果图，得到子序列级因果先验。然后，胶水层将转移频率提示和因果先验同时写入 LLM prompt，引导 SmartGen 生成目标上下文行为序列。生成后，先使用 SmartGen TOF 过滤语义不一致或无价值样本，再使用因果一致性过滤剔除违反关键因果边的样本。最后，保留的生成样本与原训练数据或目标上下文少量正常数据合并，用原 SmartGuard 训练和评估。

---

## 4 方法具体实现

### 4.1 问题定义

设智能家居设备集合为 `D`，动作集合为 `A`。一个行为表示为：

```text
b = (t, d, a), d in D, a in A
```

一个行为序列为：

```text
s = [b_1, b_2, ..., b_n]
```

原上下文为 `E_ori`，目标上下文为 `E_new`。给定原上下文正常行为序列集合 `S_ori`、设备动作集合以及上下文变化描述，目标是生成目标上下文正常行为序列 `S_gen`，并用其增广 SmartGuard 训练，使最终异常检测器在目标上下文测试集上具有更高 F1 和更低 FPR。

### 4.2 行为序列标准化

SmartGuard 与 SmartGen 源码均使用扁平四元组格式：

```text
[day, hour_slot, device_id, action_id, day, hour_slot, device_id, action_id, ...]
```

本文定义统一数据结构：

```python
BehaviorEvent(day, hour_slot, device, action)
BehaviorSequence(events)
```

胶水层只在读写 pkl 时做格式转换，不改变原项目数据文件含义。

### 4.3 事件张量桥接

GCAD 需要输入多变量时间序列。本文把离散行为序列转换为事件张量 `X`。

设通道集合为 `C`。主实验中令每个 action id 对应一个通道：

```text
C = {a_1, a_2, ..., a_m}
```

时间步使用 SmartGuard/SmartGen 中的 3 小时槽。对第 `t` 个时间槽和第 `c` 个 action 通道：

```text
X[t,c] = 1, 如果 action c 在时间槽 t 出现
X[t,c] = 0, 否则
```

也可以使用 count value 或时间衰减：

```text
X'[t] = X[t] + gamma X'[t-1]
```

这样可以缓解行为序列稀疏问题。消融实验将比较 action-level、device-level 和 device-action-level 通道粒度。

### 4.4 子序列 Granger 因果先验挖掘

在事件张量 `X in R^{T x C}` 上训练预测器：

```text
f(X_{t-lag:t-1}) -> X_t
```

预测器可以使用原 GCAD 的 TSMixer，也可以在胶水层使用轻量 MLP/Mixer 预测器以便接口测试。训练只使用原上下文正常数据。

对每个输出通道 `j`，定义通道预测损失：

```text
L_j = MSE(f_j(X_{t-lag:t-1}), X_{t,j})
```

对输入窗口求梯度并在 lag 维度求平均：

```text
A[i,j] = mean_tau | partial L_j / partial X_{t-tau,i} |
```

`A[i,j]` 表示行为通道 `i` 对行为通道 `j` 的预测影响强度。然后使用 GCAD 风格对称稀疏化：

```text
A_sparse[i,j] = max(0, A[i,j] - A[j,i]), i != j
A_sparse[i,i] = A[i,i]
A_sparse[A_sparse < h] = 0
```

最终得到正常行为因果先验 `P_causal = (A_sparse, channel_to_action, lag)`。

### 4.5 Causal-GSS Prompt Augmentation

SmartGen 原 GSS 输出的是 action transition JSON，例如某个动作后最常出现的 top-k 动作。本文在此基础上加入 causal hints：

```json
{
  "hint_type": "subsequence_granger_causality",
  "lag": 4,
  "interpretation": "source behavior tends to Granger-cause target behavior in normal routines",
  "top_causal_edges": [
    {"source": "a:14", "target": "a:15", "weight": 0.82},
    {"source": "a:15", "target": "a:16", "weight": 0.71}
  ]
}
```

Prompt 约束如下：

1. 生成目标上下文下的行为序列。
2. 不生成设备动作集合外的行为。
3. 保留与上下文无关的历史习惯。
4. 使用 SmartGen transition hints 保留高频局部转移。
5. 使用 causal hints 保留关键多步方向性关系。
6. 如果新上下文改变某类设备使用，应合理替换相关动作，而不是随机颠倒因果链。

### 4.6 生成后因果一致性过滤

SmartGen TOF 后得到候选合成序列集合 `S_candidate`。本文对每条序列计算因果覆盖率。

对 top-k 因果边 `(u -> v)`，若序列中同时出现 `u` 和 `v`，且存在 `position(u) < position(v)`，则认为该边满足；若 `v` 总是在 `u` 之前，则认为违反；若其中一个行为因目标上下文变化没有出现，则不计入检查集合。

定义：

```text
coverage(s) = sum(weight of satisfied edges) / sum(weight of checked edges)
```

当 `coverage(s) < theta` 时剔除该序列。该设计避免强制保留所有原上下文行为，同时惩罚明显颠倒关键行为链的生成样本。

### 4.7 SmartGuard 再训练与推理

过滤后的序列 `S_keep` 被写回 SmartGuard/SmartGen 的 pkl 格式。训练集可设置为：

```text
S_train = S_ori_train + S_keep
```

或在有少量目标上下文正常数据时：

```text
S_train = S_target_small + S_keep
```

然后直接调用原 SmartGuard 训练脚本。推理时仍使用 SmartGuard 的重构损失与阈值进行异常判断。本文额外保存 causal violation report，用于解释高分异常或被过滤生成样本。

### 4.8 复杂度分析

设事件张量长度为 `T`，通道数为 `C`，lag 为 `L`。轻量预测器训练复杂度近似为 `O(T * L * C * H)`，其中 `H` 为隐藏层维度。因果发现阶段需要对每个输出通道分别反向传播，复杂度约为 `O(C)` 次 backward。由于智能家居 action 数通常远小于工业传感器通道数，且该步骤离线执行，因此可接受。生成后因果过滤只遍历 top-k 因果边和序列位置，复杂度为 `O(|S_candidate| * k * n)`。

---

## 5 实验设计

本节只给出实验设置，实验结果暂不填写。

### 5.1 数据集

使用 FR、SP、US 三个真实智能家居数据集。每个数据集按上下文拆分为：

- winter 与 spring，用于季节迁移 ST；
- daytime 与 night，用于作息迁移 TT；
- single 与 multiple，用于人数迁移 NT。

原上下文为 winter/daytime/single，目标上下文为 spring/night/multiple。

### 5.2 异常类型

使用 SmartGuard 的异常注入设置：

- SD：Light flickering、Camera flickering、TV flickering。
- MD：Open window while smartlock lock、Close camera while smartlock lock。
- DM：Open air-conditioner cool mode in winter、Open window at midnight、Open watervalve at midnight。
- DD：Shower for long time、Microwave runs for long time。

### 5.3 对比方法

1. **SG-Ori**：SmartGuard 只用原上下文正常数据训练。
2. **SG-Target**：SmartGuard 用真实目标上下文正常数据训练，作为上限参考。
3. **SG+SmartGen**：使用 SmartGen 原始生成数据增广训练。
4. **SG+SmartGen+TOF**：使用 SmartGen 两阶段过滤后的生成数据增广训练。
5. **Ours**：使用 Causal-GSS prompt 与 Causal Consistency Filter 后的数据增广 SmartGuard。
6. **Ours w/o Causal-GSS**：去掉生成前 causal hints，只保留生成后因果过滤。
7. **Ours w/o Causal Filter**：只加入 causal hints，不做生成后因果过滤。
8. **Ours transition-only**：只用 SmartGen 转移频率过滤，不用 Granger 因果图。

### 5.4 评价指标

异常检测指标：

- Recall
- Precision
- F1-score
- False Positive Rate
- False Negative Rate

生成质量指标：

- Causal Coverage：生成序列满足 top-k 因果边的比例。
- Causal Violation Rate：强因果边违反比例。
- Transition Distribution Distance：生成转移分布与目标真实转移分布之间的 JS/KL 距离。
- Device/Action Shift Accuracy：目标上下文关键设备使用比例是否接近真实目标上下文。

### 5.5 实验流程

对每个数据集和每个上下文迁移类型执行以下流程：

1. 用原上下文正常训练数据运行 SmartGen 的 TSS、SSC/SPPC 和 GSS。
2. 用同一批原上下文正常训练数据构建事件张量，并挖掘因果先验。
3. 构造 Causal-GSS prompt，生成目标上下文候选序列。
4. 先执行 SmartGen TOF，再执行因果一致性过滤。
5. 用过滤后的生成序列增广 SmartGuard 训练集。
6. 在目标上下文正常测试数据和注入异常数据上评估。
7. 记录异常检测指标、生成质量指标和因果解释样例。

### 5.6 参数设置

建议默认参数：

- 事件张量粒度：action-level。
- 时间 bin：3 小时。
- lag：4。
- sparse threshold：0.001 或由验证集选择。
- top-k causal edges：20。
- causal coverage threshold：0.5。
- SmartGen 压缩阈值：沿用原论文各数据集推荐阈值。
- SmartGuard 训练参数：沿用原论文或源码默认参数。

### 5.7 消融实验

1. **通道粒度消融**：action-level、device-level、device-action-level。
2. **lag 消融**：1、2、4、8。
3. **sparse threshold 消融**：0、0.001、0.005、0.01。
4. **causal top-k 消融**：10、20、50。
5. **过滤阈值消融**：0.3、0.5、0.7。
6. **时间衰减消融**：不用衰减、0.1、0.2、0.5。

### 5.8 可解释性实验

选取被 SmartGuard 判为异常的样本，展示：

1. SmartGuard 行为级重构损失。
2. 生成或测试序列违反的 causal edges。
3. 对应行为链在原上下文正常数据中的平均出现顺序。
4. 结合设备语义解释为什么该序列异常或为什么该生成样本被过滤。

### 5.9 预期结果填写位置

实验完成后在此处填写表格：

| Dataset | Shift | Method | Precision | Recall | F1 | FPR | FNR |
|---|---|---:|---:|---:|---:|---:|---:|
| FR | ST | SG-Ori |  |  |  |  |  |
| FR | ST | SG+SmartGen |  |  |  |  |  |
| FR | ST | Ours |  |  |  |  |  |

生成质量表：

| Dataset | Shift | Method | Causal Coverage | Violation Rate | Transition Distance |
|---|---|---:|---:|---:|---:|
| FR | ST | SmartGen |  |  |  |
| FR | ST | Ours |  |  |  |

---

## 6 小结

本文初稿的核心是把三篇工作的优势通过非侵入式方式连接起来：SmartGen 解决目标上下文数据不足，GCAD-style 因果先验约束生成数据结构，SmartGuard 作为最终异常检测器。方法具体实现的关键不是修改模型，而是事件张量桥接、因果提示、因果过滤和数据再训练四个胶水模块。
