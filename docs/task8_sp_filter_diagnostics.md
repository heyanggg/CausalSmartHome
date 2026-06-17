# 任务八：SP causal filter 诊断和温和过滤复测

上一轮 SP winter -> spring 结果显示：沿用 FR 的 `k30_cov0p5_chk1/2/3` hard-deletion filter 后，SmartGen 原生 Transformer Autoencoder 的 F1 明显低于未过滤 baseline。本轮继续诊断两个问题：

1. 被删样本是否更像 spring normal test，导致误删有用样本。
2. 更温和的 filter 是否能恢复或超过未过滤 baseline。

## 1. 数据和基线

```text
dataset: sp
source context: winter
target context: spring
prior: outputs/sp_winter_to_spring_device_h0/causal_prior.json
baseline validation: outputs/sp_winter_to_spring_device_h0/smartgen_anomaly_eval_gpu/unfiltered_filter_true_vld.pkl
```

样本数：

| File | Count |
| --- | ---: |
| SmartGen `filter_true` synthetic | 140 |
| Spring normal test | 728 |
| Spring heater attack | 728 |

未过滤 baseline：

| Method | Synthetic Size | Threshold | Validation Loss Avg | Test Loss Avg | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `unfiltered_filter_true` | 140 | 0.0053 | 0.0007 | 0.5339 | 1.0000 | 0.8585 | 0.9239 |

## 2. 被删样本分布诊断

用 device/action/device-action/hour 的 Jensen-Shannon divergence 比较 `kept` 和 `rejected` 到 spring normal test 的距离。数值越小，越接近 spring normal test。

代表性结果：

| Filter | Kept | Rejected | Device JSD Kept | Device JSD Rejected | Pair JSD Kept | Pair JSD Rejected | Hour JSD Kept | Hour JSD Rejected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `k30_cov0p5_chk3` | 122 | 18 | 0.3989 | 0.5331 | 0.6235 | 0.7327 | 0.1594 | 0.2891 |
| `k10_cov0p3_chk3` | 134 | 6 | 0.3995 | 0.6331 | 0.6233 | 0.8132 | 0.1406 | 0.7464 |
| `k20_cov0p3_chk2` | 125 | 15 | 0.3922 | 0.5811 | 0.6140 | 0.8173 | 0.1447 | 0.4203 |

结论：被删样本通常比 kept 样本更远离 spring normal test。因此 SP 失败不太像“删掉的全是 spring normal 必要模式”；更像是 hard deletion 改变了训练分布和训练规模，使 Transformer Autoencoder 对 baseline validation 的重构能力崩掉。

被删样本里相对富集的模式包括：

```text
d:18|a:115
d:9|a:57
d:1|a:20
d:22|a:148
```

这些模式后续适合结合 SmartGen 设备含义表继续人工解释。

## 3. 温和过滤复测

本轮选择删除更少的四个变体：

| Slug | Raw | Kept | Rejected | Reject Ratio |
| --- | ---: | ---: | ---: | ---: |
| `k10_cov0p3_chk2` | 140 | 126 | 14 | 10.00% |
| `k10_cov0p3_chk3` | 140 | 134 | 6 | 4.29% |
| `k20_cov0p3_chk2` | 140 | 125 | 15 | 10.71% |
| `k30_cov0p3_chk3` | 140 | 128 | 12 | 8.57% |

正式命令：

```bash
/home/heyang/miniconda3/envs/smartguard_env/bin/python -m causal_smart_home.cli smartgen-anomaly-sweep-eval \
  --sweep-summary outputs/sp_winter_to_spring_device_h0/filter_sweep/filter_sweep_summary.csv \
  --out-dir outputs/sp_winter_to_spring_device_h0/smartgen_anomaly_sweep_eval_gpu_common_vld_mild \
  --dataset sp \
  --env spring \
  --select-slugs k10_cov0p3_chk2,k10_cov0p3_chk3,k20_cov0p3_chk2,k30_cov0p3_chk3 \
  --device cuda \
  --cuda-visible-devices 0 \
  --validation-pkl outputs/sp_winter_to_spring_device_h0/smartgen_anomaly_eval_gpu/unfiltered_filter_true_vld.pkl
```

结果：

| Method | Synthetic Size | Rejected | Threshold | Validation Loss Avg | Test Loss Avg | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `unfiltered_filter_true` | 140 | 0 | 0.0053 | 0.0007 | 0.5339 | 1.0000 | 0.8585 | 0.9239 |
| `k10_cov0p3_chk2` | 126 | 14 | 5.6604 | 4.5462 | 6.1197 | 0.7473 | 0.7524 | 0.7498 |
| `k10_cov0p3_chk3` | 134 | 6 | 3.2215 | 1.9021 | 4.2557 | 0.8791 | 0.6394 | 0.7403 |
| `k20_cov0p3_chk2` | 125 | 15 | 11.1368 | 9.9090 | 11.8709 | 0.8475 | 0.7790 | 0.8118 |
| `k30_cov0p3_chk3` | 128 | 12 | 4.7396 | 3.7548 | 5.1404 | 0.7555 | 0.7473 | 0.7514 |

## 4. 结论

温和过滤没有恢复 SP 性能，最佳 `k20_cov0p3_chk2` 的 F1 为 0.8118，仍显著低于未过滤 baseline 的 0.9239。

关键症状是 validation loss：

```text
unfiltered validation loss avg: 0.0007
filtered validation loss avg: 1.9021 - 9.9090
```

这意味着 filtered model 无法很好重构未过滤 baseline validation split。即使只删除 6 条样本，公共 validation loss 也会大幅上升，阈值从 0.0053 被抬到 3.2215，异常检测边界随之失真。

当前判断：

```text
SP 的问题不是简单的 filter 参数太 aggressive；
hard deletion 本身会破坏 SmartGen synthetic training distribution。
```

下一步建议从 hard deletion 转向：

1. Soft filtering / sample weighting：保留所有 140 条 synthetic sequence，用 causal coverage 调整训练采样或 loss 权重。
2. Mixture training：保留 full synthetic，同时提高高 causal-coverage 样本权重，而不是删除低 coverage 样本。
3. Data-adaptive calibration：不要只固定 FR 的 selected slugs，加入 validation compatibility 指标，例如 filtered model 对 common validation 的 loss 不应大幅高于 baseline。
4. 扩展 US winter -> spring，确认 SP 是特殊负例还是 FR 是偶然正例。
