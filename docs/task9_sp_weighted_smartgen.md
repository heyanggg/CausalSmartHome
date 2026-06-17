# 任务九：SP soft weighting SmartGen 原生异常检测

上一轮 SP 诊断显示 hard deletion 会破坏 SmartGen synthetic training distribution。本轮改为 soft weighting：

```text
保留全部 synthetic sequences
用 causal coverage 生成 sample weight
在 Transformer Autoencoder training loss 中按样本加权
```

SmartGen / SmartGuard 源码仍未改动；所有实现都在 CausalSmartHome wrapper 内。

## 1. 实现

新增 `smartgen-anomaly-eval` / `smartgen-anomaly-sweep-eval` 可选参数：

```bash
--weight-prior-json path/to/causal_prior.json
--weight-top-k-edges 30
--weight-min-edge-weight FLOAT
--weight-floor 0.2
--weight-power 1.0
```

权重公式：

```text
weight = floor + (1 - floor) * causal_coverage ** power
```

因此：

| Coverage | Weight when `floor=0.2,power=1` |
| ---: | ---: |
| 0.0 | 0.2 |
| 0.5 | 0.6 |
| 1.0 | 1.0 |

这和 hard deletion 的关键区别是：低 causal-coverage 样本仍保留，只是训练影响降低。

每次 weighted run 会额外写出：

```text
*_train_weights.pkl
*_train_weight_scores.json
```

并在 result JSON 中记录：

```text
weight_prior_json
weight_top_k_edges
weight_floor
weight_power
train_weight_min / mean / max
```

## 2. SP 权重分布

使用：

```text
prior: outputs/sp_winter_to_spring_device_h0/causal_prior.json
synthetic: sp_spring_generation_SPPC_th=0.915_gpt-4o_seq_filter_true.pkl
train split: 112
```

`top_k=30, floor=0.5, power=1` 的训练权重分布：

| Metric | Coverage | Weight | Checked Edges |
| --- | ---: | ---: | ---: |
| Min | 0.0000 | 0.5000 | 0 |
| P25 | 0.0000 | 0.5000 | 1 |
| Mean | 0.4906 | 0.7453 | 1.6429 |
| P75 | 1.0000 | 1.0000 | 3 |
| Max | 1.0000 | 1.0000 | 7 |

训练 split 中有 46/112 条 coverage 为 0，21/112 条没有命中任何因果边。

## 3. 正式训练命令

所有正式训练都使用第 0 张 RTX 3090：

```bash
/home/heyang/miniconda3/envs/smartguard_env/bin/python -m causal_smart_home.cli smartgen-anomaly-eval \
  --synthetic-pkl /home/heyang/projects/SmartGen/anomaly_detection_pipeline/synthetic_data/sp_spring_generation_SPPC_th=0.915_gpt-4o_seq_filter_true.pkl \
  --out-dir outputs/sp_winter_to_spring_device_h0/smartgen_anomaly_weighted_eval_gpu_common_vld \
  --dataset sp \
  --env spring \
  --tag weighted_k30_floor0p2_power1 \
  --device cuda \
  --cuda-visible-devices 0 \
  --validation-pkl outputs/sp_winter_to_spring_device_h0/smartgen_anomaly_eval_gpu/unfiltered_filter_true_vld.pkl \
  --weight-prior-json outputs/sp_winter_to_spring_device_h0/causal_prior.json \
  --weight-top-k-edges 30 \
  --weight-floor 0.2 \
  --weight-power 1.0
```

## 4. 结果

共同设置：

```text
baseline validation: outputs/sp_winter_to_spring_device_h0/smartgen_anomaly_eval_gpu/unfiltered_filter_true_vld.pkl
epochs: 15
split ratio: 0.8
```

| Method | Top K | Floor | Mean Weight | Threshold | Val Loss Avg | Test Loss Avg | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `unfiltered_filter_true` | - | - | - | 0.0053 | 0.0007 | 0.5339 | 1.0000 | 0.8585 | 0.9239 |
| `weighted_k30_floor0p2_power1` | 30 | 0.2 | 0.5925 | 0.0029 | 0.0009 | 0.5289 | 0.9959 | 0.8529 | 0.9189 |
| `weighted_k30_floor0p25_power1` | 30 | 0.25 | 0.6180 | 0.0047 | 0.0007 | 0.5164 | 0.9808 | 0.8551 | 0.9136 |
| `weighted_k30_floor0p3_power1` | 30 | 0.3 | 0.6434 | 0.0047 | 0.0006 | 0.5060 | 0.9753 | 0.8554 | 0.9114 |
| `weighted_k30_floor0p1_power1` | 30 | 0.1 | 0.5416 | 0.0027 | 0.0004 | 0.5106 | 0.9299 | 0.8400 | 0.8827 |
| `weighted_k30_floor0p5_power1` | 30 | 0.5 | 0.7453 | 0.0057 | 0.0007 | 0.5144 | 0.3681 | 0.6889 | 0.4799 |
| `weighted_k30_floor0p8_power1` | 30 | 0.8 | 0.8981 | 0.0091 | 0.0013 | 0.5037 | 0.3036 | 0.6481 | 0.4135 |
| `weighted_k20_floor0p2_power1` | 20 | 0.2 | 0.6050 | 0.0027 | 0.0009 | 0.4755 | 0.3558 | 0.6816 | 0.4675 |
| `weighted_k10_floor0p2_power1` | 10 | 0.2 | 0.6824 | 0.0063 | 0.0009 | 0.5133 | 0.2761 | 0.6281 | 0.3836 |
| `weighted_k30_floor0p0_power1` | 30 | 0.0 | 0.4906 | 2.4379 | 0.2749 | 0.5401 | 0.0343 | 0.2381 | 0.0600 |

## 5. 结论

Soft weighting 明显优于 hard deletion。此前最佳 hard-deletion / 温和过滤 F1 约为 0.8118，而 soft weighting 的最佳结果达到 0.9189。

不过，SP 上仍没有超过未过滤 SmartGen baseline：

```text
unfiltered baseline F1: 0.9239
best soft weighting F1: 0.9189
gap: -0.0050
```

因此当前结论应写为：

```text
For SP winter -> spring, causal soft weighting substantially reduces the damage
caused by hard deletion and nearly matches the unfiltered SmartGen baseline,
but it does not improve over the baseline under common validation calibration.
```

这说明 SP 的负例不是完全由删除样本造成的；更可能是 SP `filter_true` baseline 已经很强，而当前 causal coverage 信号主要改变 recall/precision tradeoff，没有提供额外可利用的判别收益。

下一步优先级：

1. 跑 US winter -> spring，判断 FR 正例和 SP 负例哪个更有代表性。
2. 在 FR 上回测 soft weighting，看它能否保留 FR 的提升，同时减少 hard deletion 的不稳定。
3. 若继续优化 SP，可尝试 validation-compatible weighting：训练后如果 common validation loss 或 attack recall 大幅偏移，则自动降低 weighting 强度。
