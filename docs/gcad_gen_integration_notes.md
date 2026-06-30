# GCAD 如何缝合到 Gen 中

本文对应项目主流程：

```text
causal relation prior
-> target-distribution guard
-> causal-reweighted GSS
-> Codex generation
-> Gen original two-stage TOF
-> Causal-TOF
-> Gen built-in downstream AD
-> per-seed summary
```

框架图见：[framework_diagram.svg](framework_diagram.svg)。

## 1. 理论层面

Gen 原本的核心思路是：从源上下文正常行为里提取 GSS 结构提示，再生成目标上下文正常行为，之后用 Gen 原始 TOF 和下游异常检测评估生成数据是否能帮助识别攻击。这里的 GCAD 不替换 Gen，而是作为两个“软约束注入点”接入：

1. 生成前：把 GCAD 学到的设备间方向关系变成 causal-reweighted GSS hints，影响生成器应该优先遵循哪些设备转移结构。
2. 生成后：把同一批 guarded causal hints 用在 Causal-TOF，对已经通过 Gen 原始 TOF 的序列再做因果顺序、目标分布和可选重构误差评分。

GCAD prior 的理论含义不是“物理因果真值”。项目把它定义为源上下文正常行为中的 predictive causal signal：如果设备 A 的历史窗口对预测设备 B 有明显方向性贡献，那么 A -> B 是一个可迁移的软结构证据。它只告诉生成器“这些设备共现时通常有这样的先后关系”，不能凌驾于目标上下文分布和 Gen 合法格式。

### 1.1 从 Gen 序列到 GCAD 时间序列

Gen 的数据是符号序列，每个事件是：

```text
[day, hour_slot, device_id, action_id]
```

GCAD/causal-relation 类方法通常需要多变量时间序列 `[T, C]`。因此项目先用 `EventTensorizer` 把离散事件投影到周内 3 小时时间槽：

```text
channel = device / action / device_action
time = day * 8 + hour_slot
value = count 或 binary
```

主实验使用 device-level channel，例如 `d:13`。这样 causal matrix 的行列都能和 Gen 的设备 ID 对齐。

### 1.2 GCAD prior 的方向强度

本地 fallback `GradientCausalMiner` 的思路是：

1. 用过去 `lag` 个时间步预测当前每个 channel。
2. 对每个输出 channel 单独计算预测损失。
3. 将该输出损失反向传播到输入窗口。
4. 对输入 channel 的绝对梯度在样本和 lag 维度取平均。
5. 得到输入 channel 对输出 channel 的影响矩阵。
6. 用 `matrix - matrix.T` 保留方向不对称部分，并归一化、稀疏化。

因此矩阵元素 `M[i, j]` 可以理解为：源正常行为中，channel `i` 对预测 channel `j` 的方向性贡献强度。项目随后把非零边抽成：

```text
source, target, source_index, target_index, weight, lag
```

### 1.3 为什么需要 target-distribution guard

GCAD prior 来自源上下文，目标是生成目标上下文正常行为。源上下文里强的因果边，可能指向目标上下文里很少出现甚至不应该频繁出现的设备。如果直接把这些边喂给生成器，生成数据会偏向源分布。

所以项目先计算：

```text
source/prompt observed device distribution
target normal device distribution
overuse_ratio = observed_freq / max(target_freq, min_target_freq, eps)
```

当边的端点设备在目标分布中过度使用时，guard 会：

```text
suppress: guarded_weight = 0
downweight: guarded_weight = raw_weight * downweight_factor
```

README 固定主流程里默认 `guard-mode=downweight`。这保留了“这条边有 GCAD 证据”的可解释性，同时降低它对目标生成的支配力。

### 1.4 GCAD 与 Gen GSS 的融合公式

Gen GSS 的结构来自源正常序列中的相邻设备转移：

```text
transition_score(A -> B) = count(A followed by B) / outgoing_count(A)
```

GCAD 给的是因果方向强度：

```text
guarded_causal_strength(A -> B)
```

项目先分别归一化：

```text
normalized_transition = transition_score / max_transition
normalized_causal = guarded_causal_strength / max_guarded_causal_strength
```

然后支持两种融合：

```text
additive:
final_score = normalized_transition + lambda_causal * normalized_causal

multiplicative:
final_score = normalized_transition * (1 + lambda_causal * normalized_causal)
```

主流程默认 multiplicative，并且默认 `add_causal_edges=True`。含义是：

1. 如果一条边既是 Gen GSS 常见转移，又有 GCAD 支持，它会被提高排序。
2. 如果一条 GCAD 边没有出现在相邻转移里，但通过 guard 后仍有强度，可以作为 causal_relation_augmented edge 加入提示，避免只因 Gen GSS 只统计相邻事件而丢掉长程结构。
3. 如果 raw causal relation 和 guarded reweighted hints 冲突，生成提示要求以后者为准。

### 1.5 Causal-TOF 的理论位置

Gen original TOF 已经完成两类筛选：

1. reconstruction-loss outlier detection
2. utility/value selection

Causal-TOF 不是另起炉灶的方法，而是主 pipeline 的后处理评分步骤。它对通过 Gen TOF 的每条序列计算：

```text
final_score =
    alpha_rec * reconstruction_loss
  + beta_violation * causal_violation
  + gamma_dist * distribution_penalty
```

其中：

```text
causal_violation = violated_weight / (satisfied_weight + violated_weight)
sample_weight = exp(-temperature * final_score)
```

边满足的判定是：生成序列里至少存在一个 source 位置早于一个 target 位置。缺失 source 或 target 会进入 audit 的 missing 字段。默认情况下，`guard_action=downweight` 的边保留在诊断字段里，但不计入 violation penalty；只有显式开启 `--penalize-downweighted-edges` 才把它们纳入惩罚。

## 2. 代码层面

### 2.1 数据结构入口

`causal_smart_home/schema.py`

定义 `BehaviorEvent` 和 `BehaviorSequence`：

```text
BehaviorEvent(day, hour_slot, device, action)
BehaviorSequence(events, sequence_id, meta)
```

关键方法：

```text
BehaviorSequence.from_flat_numeric(flat)
BehaviorSequence.to_flat_numeric()
BehaviorEvent.key("device") -> d:<id>
BehaviorEvent.key("action") -> a:<id>
BehaviorEvent.key("device_action") -> d:<id>|a:<id>
```

这些方法保证 CausalSmartHome 内部可以用对象表达逻辑，但和 Gen pickle 交互时仍然输出原始 flat quadruples。

### 2.2 序列张量化

`causal_smart_home/event_tensor.py`

`EventTensorizer.fit_transform(sequences)` 完成：

```text
统计出现频次 -> 选 channel -> 建 [bin_count, channel_count] 数组 -> 写入事件 -> 可选 decay
```

主流程在 `causal_relation_prior_source.py` 中调用：

```python
EventTensorizer(level=level, count_mode="binary", decay=0.1).fit_transform(sequences)
```

默认 `level=device`，因此输出 channels 是 `d:<device_id>`，后续可以直接和 GSS device transition 对齐。

### 2.3 prior 解析与统一

`causal_smart_home/causal_relation_prior_source.py`

入口函数：

```python
resolve_causal_relation_prior(
    prior_json=None,
    prior_matrix_path=None,
    source_pkl=None,
    out_dir=None,
    adapter_mode="existing",
    level="device",
    lag=4,
    sparse_threshold=0.001,
    seed=2024,
)
```

优先级：

```text
prior_json -> prior_matrix_path -> source_pkl via adapter
```

不管来源是什么，最终都返回 `ResolvedCausalRelationPrior`：

```text
causal_relation_source
level
lag
sparse_threshold
channels
matrix
top_causal_edges
config
meta
```

这样后续代码不需要关心 prior 来自外部 GCAD JSON、矩阵文件，还是本地 fallback miner。

如果走 `source_pkl`，代码路径是：

```text
_load_behavior_sequences_from_pickle
-> EventTensorizer
-> CausalRelationAdapter.mine_event_prior
-> GradientCausalMiner.fit_prior
-> _standardize_edges(prior.top_edges(...))
```

### 2.4 GCAD fallback miner

`causal_smart_home/causal_prior.py`

核心类：

```text
GradientCausalMiner
CausalPrior
```

`fit_predictor()` 先用 `_windows()` 构造监督样本：

```text
xs[t] = x[t-lag:t]
ys[t] = x[t]
```

再训练 `_TinyMixer`。`discover()` 对每个输出 channel 计算：

```python
loss = mean((pred[:, out_ch] - y[:, out_ch]) ** 2)
grad = autograd.grad(loss, xb)
g = mean(abs(grad), dim=(sample, lag))
```

最后 `sparsify()` 做方向差分：

```python
diff = matrix - matrix.T
out = maximum(diff, 0)
out /= max_positive
out[out < threshold] = 0
```

### 2.5 target guard

`causal_smart_home/target_distribution_guard.py`

入口：

```python
compute_device_distribution(sequences)
apply_target_distribution_guard(causal_edges, generated_or_prompt_distribution, target_distribution, config)
```

`apply_target_distribution_guard()` 对每条边：

1. 提取 source/target device key。
2. 计算 source/target 在 observed 与 target 中的频率。
3. 根据 `endpoint_policy` 判断是否 guard。
4. 根据 `mode` 写入：

```text
raw_weight
guarded_weight
weight
guard_action
guard_reason
source_overused / target_overused
source_observed_freq / target_observed_freq
source_target_freq / target_target_freq
source_overuse_ratio / target_overuse_ratio
```

输出一份 guarded edges 和一份 guard report，后者用于审计和 prompt package。

### 2.6 GSS 重加权

`causal_smart_home/causal_gss_reweight.py`

`build_device_transition_graph(sequences)` 从源正常序列统计相邻设备转移：

```text
counts[(source, target)] += 1
outgoing[source] += 1
transition_score = count / outgoing[source]
```

`reweight_gss_edges()` 将 transition edges 和 guarded causal edges 对齐：

```text
pair = (canonical_source_device, canonical_target_device)
```

然后 `_score_edge_row()` 生成最终 hint row：

```text
source_device
target_device
source_device_key
target_device_key
transition_score
normalized_transition_score
raw_causal_strength
guarded_causal_strength
normalized_guarded_causal_strength
final_score
edge_origin
guard_action
guard_reason
lag
```

`edge_origin` 有两类：

```text
transition_existing
causal_relation_augmented
```

这正是“把 GCAD 缝到 Gen GSS 里”的代码位置。

### 2.7 prompt 构建

`scripts/build_causal_gss_prompt.py`

完整步骤：

```text
读 source/target pkl
-> resolve_causal_relation_prior
-> build_device_transition_graph(source)
-> compute_device_distribution(source/target)
-> apply_target_distribution_guard
-> reweight_gss_edges
-> 写 prompt.txt / resolved prior / guard report / guarded hints / config
```

`build_prompt_text()` 明确设置提示优先级：

1. guarded causal-reweighted GSS hints 是主要结构。
2. raw causal relation hints 只是弱背景。
3. 如果 raw causal relation 与 guarded hints 冲突，遵循 guarded hints。
4. 不要过度生成 guard report 中标记过度使用的设备。
5. causal relation edges 不是物理真因果，只是源上下文预测性因果信号。

### 2.8 Codex 生成与打包

`scripts/build_codex_generation_package.py`

把 prompt 相关文件复制成自包含 generation package。

`scripts/validate_and_pack_codex_generation.py`

校验 Codex JSONL：

```text
sequence 字段存在
长度非空且是 4 的倍数
day in 0..6
hour_slot in 0..7
device_id 在 dictionary
action_id 在 dictionary
device/action 搭配合法
数量等于 expected-count
长度等于 expected-length（如果指定）
```

通过后写成 Gen 可读 pickle。

### 2.9 Gen original TOF

`causal_smart_home/gen_original_tof.py`

这个模块不重写 Gen TOF，只做封装：

```text
generated pkl
-> copy 到 Gen filter_data 期望路径
-> import/call security_check.py
-> 优先取 *_filter_true.pkl
-> fallback 到 *_filter.pkl 或 input
-> 写 gen_original_tof_report.json/md
```

对应 CLI 是 `scripts/run_gen_original_tof.py`。

### 2.10 Causal-TOF

`causal_smart_home/causal_tof.py`

核心函数：

```python
score_sequence_causal_tof(...)
score_sequences_causal_tof(...)
weighted_resample_sequences(...)
```

`score_sequence_causal_tof()` 对每条 guarded edge：

1. 取边权重。
2. 找 source/target 在序列中的所有位置。
3. 如果至少一组 `source_pos < target_pos`，进入 satisfied。
4. 如果 source/target 都出现但没有合法顺序，进入 violated。
5. 如果 source 或 target 缺失，进入 missing。

再计算：

```text
causal_coverage
causal_violation
observed_causal_coverage_all_guarded_edges
observed_causal_violation_all_guarded_edges
distribution_penalty
final_score
sample_weight
```

`scripts/run_causal_tof.py` 根据 `--mode` 输出：

```text
rank: 只排序
weight: 按 sample_weight 加权重采样
filter: sample_weight < min_weight 的序列删除
```

### 2.11 Gen downstream AD

`causal_smart_home/gen_downstream_ad.py`

导入 vendored Gen `anomaly_detection_pipeline/models1.py`，并保持 Gen 的评估协议：

```text
synthetic normal pkl
-> train TransformerAutoencoder
-> validation reconstruction loss percentile gives threshold
-> target normal test + attack test
-> TP/TN/FP/FN, precision, recall, F1, FPR/FNR
```

`multiple` 场景特殊：Gen 原协议使用完整 filtered synthetic set 同时做训练和 threshold calibration；代码在 `_prepare_train_validation_files()` 中保留这个行为。

### 2.12 汇总

`scripts/run_gen_downstream_ad.py`

负责包装单次 AD 运行，并把 raw payload 归一化为：

```text
dataset, scenario, seed, variant
input_stage
used_gen_original_tof
used_causal_tof
num_generated_before_tof
num_generated_after_gen_tof
num_generated_after_causal_tof
precision, recall, f1, accuracy, fpr, fnr
```

`scripts/summarize_main_experiment.py` 只保留：

```text
ablation_no_causal_tof
proposed_causal_gss_codex_causal_tof
```

并写出 per-seed CSV/JSON/Markdown。README 也强调主结果必须按 seed 展示，不用 mean/std 或 delta 表替代。

## 3. 最短代码调用链

```text
scripts/build_causal_gss_prompt.py
  -> causal_relation_prior_source.resolve_causal_relation_prior
     -> EventTensorizer
     -> CausalRelationAdapter
     -> GradientCausalMiner
  -> target_distribution_guard.apply_target_distribution_guard
  -> causal_gss_reweight.build_device_transition_graph
  -> causal_gss_reweight.reweight_gss_edges

scripts/validate_and_pack_codex_generation.py
  -> schema.load_numeric_sequences

scripts/run_gen_original_tof.py
  -> gen_original_tof.run_gen_original_tof
     -> vendored gen_core/gen_original_tof/security_check.py

scripts/run_causal_tof.py
  -> causal_tof.extract_guarded_edges
  -> causal_tof.score_sequences_causal_tof
  -> causal_tof.weighted_resample_sequences 或 filter

scripts/run_gen_downstream_ad.py
  -> gen_downstream_ad.run_gen_downstream_ad_experiment
     -> vendored gen_core/anomaly_detection_pipeline/models1.py

scripts/summarize_main_experiment.py
  -> collect_per_seed_rows
  -> write_outputs
```

## 4. 一句话总结

这个项目把 GCAD 当成“可审计的软结构先验”，而不是替换 Gen 的生成或评估算法：生成前，GCAD 通过 guard 和 GSS 重加权影响提示；生成后，GCAD 通过 Causal-TOF 影响序列保留/重采样；最终评价仍走 Gen 原始 TOF 和 Gen built-in downstream AD。
