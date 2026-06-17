# 任务十：FR soft weighting 回测

本轮目标是用任务九新增的 SmartGen causal soft weighting，在 FR winter -> spring 上回测：

```text
能否保留 FR hard deletion 的提升，同时减少 SP 上 hard deletion 的不稳定？
```

结论先行：

```text
FR soft weighting 能超过未过滤 baseline，但没有超过 hard deletion 最佳结果。
```

## 1. 对照设置

```text
dataset: fr
source context: winter
target context: spring
prior: outputs/fr_winter_to_spring_device_h5e-05/causal_prior.json
synthetic: /home/heyang/projects/SmartGen/anomaly_detection_pipeline/synthetic_data/fr_spring_generation_SPPC_th=0.918_gpt-4o_seq_filter_true.pkl
baseline validation: outputs/fr_winter_to_spring_device_h5e-05/smartgen_anomaly_eval_gpu/unfiltered_filter_true_vld.pkl
device: cuda
cuda-visible-devices: 0
```

未过滤 baseline 和 hard-deletion 对照来自前序 FR 正式实验：

| Method | Synthetic Size | Threshold | Val Loss Avg | Test Loss Avg | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `unfiltered_filter_true` | 125 | 1.3921 | 0.1578 | 3.0830 | 0.9886 | 0.7311 | 0.8406 |
| `hard_k30_cov0p5_chk1` | 109 | 1.6313 | 0.3428 | 2.6629 | 0.9886 | 0.7768 | 0.8700 |
| `hard_k30_cov0p5_chk2` | 118 | 0.0136 | 0.0848 | 2.7317 | 1.0000 | 0.6822 | 0.8111 |
| `hard_k30_cov0p5_chk3` | 121 | 0.0066 | 0.0700 | 2.7571 | 1.0000 | 0.6822 | 0.8111 |

## 2. Weighted 训练命令

代表命令：

```bash
/home/heyang/miniconda3/envs/smartguard_env/bin/python -m causal_smart_home.cli smartgen-anomaly-eval \
  --synthetic-pkl /home/heyang/projects/SmartGen/anomaly_detection_pipeline/synthetic_data/fr_spring_generation_SPPC_th=0.918_gpt-4o_seq_filter_true.pkl \
  --out-dir outputs/fr_winter_to_spring_device_h5e-05/smartgen_anomaly_weighted_eval_gpu_common_vld \
  --dataset fr \
  --env spring \
  --tag weighted_k30_floor0p2_power1 \
  --device cuda \
  --cuda-visible-devices 0 \
  --validation-pkl outputs/fr_winter_to_spring_device_h5e-05/smartgen_anomaly_eval_gpu/unfiltered_filter_true_vld.pkl \
  --weight-prior-json outputs/fr_winter_to_spring_device_h5e-05/causal_prior.json \
  --weight-top-k-edges 30 \
  --weight-floor 0.2 \
  --weight-power 1.0
```

## 3. Weighted 结果

| Method | Top K | Floor | Mean Weight | Threshold | Val Loss Avg | Test Loss Avg | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `weighted_k30_floor0p1_power1` | 30 | 0.1 | 0.8830 | 1.8726 | 0.1791 | 3.7833 | 0.9886 | 0.7500 | 0.8529 |
| `weighted_k30_floor0p2_power1` | 30 | 0.2 | 0.8960 | 2.0219 | 0.1953 | 3.8349 | 0.9886 | 0.7565 | 0.8571 |
| `weighted_k30_floor0p3_power1` | 30 | 0.3 | 0.9090 | 1.9316 | 0.1837 | 3.7975 | 0.9886 | 0.7565 | 0.8571 |
| `weighted_k30_floor0p5_power1` | 30 | 0.5 | 0.9350 | 2.0448 | 0.1798 | 3.8054 | 0.9886 | 0.7565 | 0.8571 |
| `weighted_k10_floor0p2_power1` | 10 | 0.2 | 0.8960 | 2.0219 | 0.1953 | 3.8349 | 0.9886 | 0.7565 | 0.8571 |
| `weighted_k20_floor0p2_power1` | 20 | 0.2 | 0.8960 | 2.0219 | 0.1953 | 3.8349 | 0.9886 | 0.7565 | 0.8571 |

FR 上 `top_k=10/20/30` 的 `floor=0.2` 结果完全一致，说明 FR train split 中实际影响权重的 causal coverage 在这些 top-k 设置下没有区别。

## 4. 与 SP 对照

| Dataset | Method | Best F1 | Compared To Baseline |
| --- | --- | ---: | ---: |
| FR | unfiltered baseline | 0.8406 | - |
| FR | hard deletion | 0.8700 | +0.0294 |
| FR | soft weighting | 0.8571 | +0.0165 |
| SP | unfiltered baseline | 0.9239 | - |
| SP | hard deletion / mild filtering | 0.8118 | -0.1121 |
| SP | soft weighting | 0.9189 | -0.0050 |

## 5. 结论

Soft weighting 的定位更清楚了：

```text
它比 hard deletion 稳定，能避免 SP 上的大幅崩坏；
但在 FR 正例上，它的收益小于 hard deletion。
```

因此，当前主线不应简单替换为 soft weighting，也不应继续只押 hard deletion。更稳妥的实验策略是：

1. FR 报告 hard deletion 是当前最佳正例，同时补充 soft weighting 作为稳定性更好的折中方案。
2. SP 报告 soft weighting 大幅缓解 hard deletion 负效应，但未超过未过滤 baseline。
3. 下一步跑 US winter -> spring，形成 FR/SP/US 三数据集结论。
4. 如果要继续方法改进，应做 hybrid selector：当 hard deletion 过滤后 common validation compatibility 变差时，自动退回 soft weighting。
