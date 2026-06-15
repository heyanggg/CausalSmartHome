# 任务一：三篇论文与三份源码梳理报告

## 0. 阅读边界与结论

本报告只围绕 SmartGuard、SmartGen 与 GCAD 三个项目。压缩包中另外一个失败实验目录没有被用于方案、论文初稿或代码实现；本项目只把它视为负面经验：不要把三者的模型主体混在一起重写，而要在数据、提示词、过滤、评估这些接口位置做胶水层。

三个项目的共同点是：它们都围绕“序列中的依赖结构是否正常”展开，只是数据形态不同。SmartGuard 面向离散智能家居行为序列，核心是时间感知的自编码异常检测；SmartGen 面向上下文变化后的行为序列生成，核心是把历史行为压缩、抽取转移提示，再用 LLM 生成目标上下文数据；GCAD 面向连续多变量时间序列，核心是用预测器梯度挖掘动态 Granger 因果图并用因果偏离检测异常。三者可缝合的关键接口不是模型层，而是“离散行为序列 -> 多变量事件时间序列 -> 因果先验 -> 生成提示和过滤 -> SmartGuard 训练数据”。

---

## 1. SmartGuard 论文与源码梳理

### 1.1 研究问题

SmartGuard 研究智能家居用户行为序列的无监督异常检测。行为定义为三元组 `b=(t,d,c)`，其中 `t` 是时间戳，`d` 是设备，`c` 是设备控制；行为序列是按时间排序的行为列表。任务是在只使用正常样本训练的前提下，判断一个行为序列是否异常。

论文把异常分为四类：

1. SD：单设备上下文异常，例如灯或摄像头高频闪烁。
2. MD：多设备组合异常，例如门锁上锁时打开窗户或关闭摄像头。
3. DM：设备控制-时刻异常，例如冬天打开空调制冷、午夜打开水阀。
4. DD：设备控制-持续时间异常，例如水阀或微波炉运行过久。

### 1.2 论文核心方法

SmartGuard 是 Transformer Autoencoder 结构，训练时重构正常行为序列，测试时用重构损失判断异常。三个关键组件是：

**LDMS：Loss-guided Dynamic Mask Strategy。** 普通自编码器容易学好高频行为、忽略低频但正常的行为。LDMS 先让模型无 mask 训练若干 epoch 以稳定收敛，然后根据上一轮每个行为的重构损失，把高损失行为作为更需要学习的对象进行动态 mask，迫使模型学习 hard-to-learn 行为。

**TTPE：Three-level Time-aware Position Embedding。** 论文把行为的时间上下文拆为 order-level、moment-level 和 duration-level：序列顺序、星期/小时、同一设备下一次操作之间的持续时间。最终位置编码为这几类时间编码的加权和，再与设备控制 embedding 相加。

**NWRL：Noise-aware Weighted Reconstruction Loss。** 家居行为存在噪声行为，如用户临时播放音频或设备被动刷新。SmartGuard 用训练后的行为损失向量估计噪声行为权重，对高损失潜在噪声行为降低权重，避免正常序列因少量噪声而被误报。

### 1.3 SmartGuard 源码结构

压缩包中的 SmartGuard 主要文件：

- `SmartGuard/SmartGuard.py`：模型主体。包含 `TimeSeriesDataset`、`PegasusSinusoidalPositionalEmbedding` 和 `SmartGuard` 类。`TimeSeriesDataset` 假设每个样本是长度 40 的扁平序列，即 10 个行为，每个行为四元组为 `[day, hour, device, action]`。模型内部用 `sample.reshape(10,4).T[3]` 提取 action/control 序列作为重构目标。
- `SmartGuard/train.py`：训练、验证、阈值、行为权重、攻击评估全部在一个脚本中。核心流程是训练 SmartGuard，保存 `saved_model/final1_...pth`，再用验证集分位数确定阈值，并按 SD/MD/DM/DD 攻击文件评估。
- `SmartGuard/evaluate_smartguard.py`：对原训练脚本做了评估入口封装，可以通过 `--model_path` 指定模型权重。
- `SmartGuard/other_baseline.py`：传统和深度基线。
- `SmartGuard/data/data/{an,fr,sp}/dictionary.py`：设备与动作字典。

### 1.4 源码输入输出与限制

SmartGuard 对数据格式依赖较强：默认每个序列 10 个行为、长度 40；每个行为四元组 `[day, hour_slot, device_id, action_id]`。这给胶水层提供了明确接口：只要生成或过滤后的行为序列仍保持这个格式，就可以不改 SmartGuard 模型代码。

主要限制：

- 原始代码的路径写死较多，例如 `data/{dataset}_data/...`。
- `vocab_dic` 在 SmartGuard 与 SmartGen 中数值略有差异，例如 FR/SP 是否包含 padding id 需要对齐。
- 训练脚本把训练、阈值、评估耦合较紧，适合用 subprocess 包一层，而不是重写内部逻辑。

---

## 2. SmartGen 论文与源码梳理

### 2.1 研究问题

SmartGen 解决智能家居模型在行为漂移下的自适应问题。用户行为会因季节、作息、入住人数变化而变化；持续采集新数据成本高、速度慢、涉及隐私。SmartGen 让 LLM 根据原上下文行为数据和新上下文描述，生成目标上下文的正常行为序列，用于下游异常检测和行为预测模型的再训练。

论文考虑三类上下文迁移：

1. ST：季节迁移，例如 winter -> spring。
2. TT：作息迁移，例如 daytime -> night。
3. NT：人数迁移，例如 single -> multiple。

### 2.2 论文核心方法

SmartGen 由四个模块组成：

**TSS：Time and Semantic-aware Split。** 长行为序列没有自然标点，直接喂给 LLM 会过长且语义混乱。TSS 用时间间隔阈值和总时长阈值切分，同时用语义检查避免把相关行为拆开。

**SSC：Semantic-aware Sequence Compression。** 通过 Transformer Autoencoder 把行为序列映射到语义空间，再用余弦相似度去冗余，保留代表性序列，降低 LLM 输入长度。源码中该模块早期命名为 SPPC。

**GSS：Graph-guided Sequence Synthesis。** 从原始序列构建行为转移图和 action transition matrix，选取 top-k 高频转移，保存为 JSON hints，作为 LLM 生成时的用户习惯提示。

**TOF：Two-stage Outlier Filter。** 第一阶段用重构损失和 IQR 找出可疑生成序列；第二阶段把可疑序列逐个加入训练集，若能降低验证损失则保留，否则剔除。

### 2.3 SmartGen 源码结构

压缩包中的 SmartGen 主要文件：

- `SmartGen/SmartGen/main.py`：总入口。流程为 `Split -> Dayse -> Train/SPPC_select -> Find_categories -> ATM -> Transtext -> LLM_call -> Extract -> Transnum -> security_check -> downstream test`。其中 `need_generate` 控制是否生成，`need_test` 控制是否做下游任务。
- `SmartGen/SmartGen/split.py`：TSS 实现。`calculate_hours` 按 3 小时槽计算时间间隔；`split` 先按相邻间隔阈值 9 小时切，再按累计时长阈值 24 小时切；`semantic_judge` 用 off-action 字典避免把相关动作拆开。
- `SmartGen/SmartGen/dayse.py`：按星期几分类，得到 `trn_day_{day}.pkl`。
- `SmartGen/SmartGen/baseline2.py`：训练 TransformerAutoencoder，用于 SSC/SPPC 的语义表示。
- `SmartGen/SmartGen/sppc.py`：`SPPC_select` 加载训练好的 TransformerAutoencoder，取 encoder/decoder 输出展平后计算余弦相似度，用阈值去重；`similarity_select` 是不经过语义模型的直接相似度去重。
- `SmartGen/SmartGen/text_translation_matrix.py`：GSS 的转移图和 JSON transition hints。核心类 `LinkAnalyzer` 统计每个 action 后续 action 的 top-k 次数。
- `SmartGen/SmartGen/transtext.py`、`transnumber.py`、`extract.py`：数字序列与文本序列之间转换，解析 LLM 输出。
- `SmartGen/SmartGen/security_check.py`：TOF 实现。先训练自编码器，计算生成序列重构损失，IQR 过滤；再对 outlier 逐个做价值评估。
- `anomaly_detection_pipeline` 与 `behavior_prediciton_pipeline`：下游任务再训练与评估。

### 2.4 源码输入输出与限制

SmartGen 与 SmartGuard 的行为格式非常接近，都是扁平四元组序列。差异是 SmartGen 的生成阶段会把数字四元组翻译为文本，再把 LLM 输出解析回数字序列。

主要限制：

- `main.py` 里的 LLM API key/base_url 留空，无法直接在线生成。
- 路径耦合较强，目录结构需符合 `IoT_data/{dataset}/{env}/...`。
- GSS 本质是局部转移频率提示，尚未显式表达多步、滞后、非对称因果关系。这正是可缝合 GCAD 的入口。

---

## 3. GCAD 论文与源码梳理

### 3.1 研究问题

GCAD 研究多变量时间序列异常检测。传统 GNN 或自适应图方法多通过预测/重构误差间接学习变量关系，图结构往往是 embedding 相似度，不一定解释变量如何影响时间序列演化。GCAD 的假设是：异常发生时，变量之间的动态 Granger 因果模式会显著偏离正常模式。

### 3.2 论文核心方法

GCAD 包含四个模块：

**Prediction-based Gradient Generator。** 用 TSMixer/RevIN 风格预测器在正常数据上学习下一时刻预测。

**Granger Causality Discovery。** 对每个输出通道的预测误差单独反向传播，得到输入窗口各变量对该输出变量误差的梯度。对时间滞后维度的绝对梯度积分，作为变量 i 对变量 j 的 Granger 因果强度。

**Causality Graph Sparsification。** 因果应是方向性的，而相似性常是对称的。GCAD 用 `A - A^T` 的思想去掉双向对称成分，保留非对称方向强度，再用阈值去除弱边。

**Causal Deviation Scoring。** 从正常训练窗口采样得到典型正常因果图 `A_norm`；测试时计算每个窗口因果图与 `A_norm` 的相对偏差，并结合主对角线的时间模式偏差得到异常分数。

### 3.3 GCAD 源码结构

压缩包中的 GCAD 主要文件：

- `GCAD/main.py`：训练入口。加载数据、训练 `TSMixerRevIN`，保存 best model；训练结束后调用 `save_train_mean_causal` 保存正常因果图，再调用 `test` 做评估。
- `GCAD/models/tsmixer.py`：预测器主体，包含 RevIN、ResBlock、TSMixerRevIN。
- `GCAD/test.py`：因果发现与异常评分核心。`save_train_mean_causal` 中对每个输出 feature 的 loss 分别反向传播，得到 `grad_causal_mat`；再按 GCAD 稀疏化策略处理并保存均值因果图。`test` 用同样方式计算测试窗口因果图，和保存的标准因果图做相对误差，最后输出 ROC、PRC、F1 等指标。
- `GCAD/utils/dataloader.py`：读取 `train.csv` 和 `test.csv`，默认最后一列为 label，其他列为多变量时间序列通道。
- `GCAD/utils/general.py`：随机种子设置。

### 3.4 源码输入输出与限制

GCAD 接受连续时间序列 CSV：`train.csv` 与 `test.csv`。这与 SmartGuard/SmartGen 的离散行为序列不同，因此不能直接套用。合理胶水做法是：把每个 action、device 或 device-action motif 转为多变量事件时间序列通道，例如每 3 小时一个 bin，通道值表示该 bin 中动作是否发生或发生次数。这样 GCAD 不需要改代码/思想，就能在行为序列上挖掘“行为子序列因果图”。

主要限制：

- 原 GCAD 代码是实验脚本，不是库 API，适合用准备好的 CSV 目录调用，或在胶水层复刻一个小型 GCAD-style miner 用于事件张量。
- 如果直接把行为 id 当连续值喂给 GCAD，会偏离含义；必须先 one-hot/count/binary tensorize。
- GCAD 当前主要建模二元因果对，不能直接处理多变量组合因果；胶水层可以先用 action/device/motif channel 近似。

---

## 4. 三者接口对应关系

| 维度 | SmartGuard | SmartGen | GCAD | 胶水层连接方式 |
|---|---|---|---|---|
| 数据对象 | 行为序列 `[day,hour,device,action]` | 同样的行为序列，另有文本化表示 | 多变量时间序列 `[T,C]` | EventTensorizer：action/device/motif -> channel |
| 主要能力 | 异常检测 | 生成目标上下文正常数据 | 挖掘动态因果偏离 | 用因果先验指导生成和过滤 |
| 不应改动处 | LDMS/TTPE/NWRL 模型 | TSS/SSC/GSS/TOF 工作流 | TSMixer/gradient causal discovery | 只在输入输出边界做 wrapper |
| 输出 | 异常分数/标签 | 合成行为序列 | 因果矩阵/异常分数 | 合成数据质量增强与解释 |

---

## 5. 对缝合的直接启示

1. SmartGuard 与 SmartGen 的数据格式天然兼容，都是智能家居行为四元组序列。因此 SmartGen 生成的目标上下文数据可以直接用于 SmartGuard 再训练，只需保证 padding、vocab 和路径一致。
2. SmartGen 的 GSS 已经有行为转移图，但它是频次图，不是因果图；GCAD 提供了把“预测影响”转为有向图的机制，可以补足 GSS 的因果约束。
3. GCAD 不能直接处理离散行为 id，必须通过事件张量桥接。推荐通道粒度从 action 开始，再做 device 与 device-action ablation。
4. 最稳定的胶水位置有两个：LLM prompt 前加入 causal hints；LLM 输出后加入 causal consistency filter。这样不用修改 SmartGen 和 SmartGuard 的内部模型，也不会把三个项目魔改成一个不可控的新模型。
5. 主实验应评估“因果增强生成数据是否让 SmartGuard 在上下文漂移下更少误报、更稳检测攻击”，而不是单独证明 GCAD 在智能家居数据上比 SmartGuard 强。
