# 任务七：SP winter -> spring SmartGen 原生异常检测扩展

本轮目标是把 FR winter -> spring 已跑通的 SmartGen 原生异常检测流程扩展到 SP winter -> spring：

```text
build prior -> causal filter sweep -> GPU unfiltered baseline
-> baseline validation common calibration -> GPU causal-filtered variants
-> summarize F1 / Recall / Precision / threshold
```

所有训练命令都在 CausalSmartHome wrapper 内执行，并显式使用：

```bash
--device cuda --cuda-visible-devices 0
```

SmartGen / SmartGuard 源码没有改动，模型、split、summary 和中间 pkl 均写入 CausalSmartHome 的 `outputs/`。

## 1. 数据和输出目录

```text
dataset: sp
source context: winter
target context: spring
SmartGen synthetic: /home/heyang/projects/SmartGen/anomaly_detection_pipeline/synthetic_data/sp_spring_generation_SPPC_th=0.915_gpt-4o_seq_filter_true.pkl
normal test: /home/heyang/projects/SmartGen/anomaly_detection_pipeline/test/sp/spring/test.pkl
attack: /home/heyang/projects/SmartGen/anomaly_detection_pipeline/attack/sp/labeled_sp_spring_attack_heater.pkl
```

样本数：

| Split / File | Count |
| --- | ---: |
| SP winter `split_trn.pkl` | 6283 |
| SP spring SmartGen `filter_true` synthetic | 140 |
| SP spring normal test | 728 |
| SP spring heater attack | 728 |

主输出目录：

```text
outputs/sp_winter_to_spring_device_h0/
```

说明：先按 FR 的 `sparse-threshold=5e-05` 试跑了 `outputs/sp_winter_to_spring_device_h5e-05/`，但 SP 的 device-level prior 被 sparsify 后非零边为 0，filter sweep 全部退化为 `kept=140/rejected=0`。因此 SP 主实验改用 `sparse-threshold=0`，保留弱因果边用于过滤。

## 2. Causal prior 和 filter sweep

主 prior：

```bash
/home/heyang/miniconda3/envs/smartguard_env/bin/python -m causal_smart_home.cli build-prior \
  --train-pkl /home/heyang/projects/SmartGen/SmartGen/IoT_data/sp/winter/split_trn.pkl \
  --out-dir outputs/sp_winter_to_spring_device_h0 \
  --lag 4 \
  --epochs 40 \
  --level device \
  --sparse-threshold 0
```

Prior 摘要：

| Item | Value |
| --- | ---: |
| Device channels | 19 |
| Non-zero matrix entries | 190 |
| Final train loss | 0.001243 |

Filter sweep：

```bash
/home/heyang/miniconda3/envs/smartguard_env/bin/python -m causal_smart_home.cli sweep-filter \
  --prior-json outputs/sp_winter_to_spring_device_h0/causal_prior.json \
  --generated-pkl /home/heyang/projects/SmartGen/anomaly_detection_pipeline/synthetic_data/sp_spring_generation_SPPC_th=0.915_gpt-4o_seq_filter_true.pkl \
  --out-dir outputs/sp_winter_to_spring_device_h0/filter_sweep \
  --tag sp_spring_filter_true \
  --top-k-edges 10,20,30 \
  --min-coverages 0.3,0.5,0.7 \
  --min-checked-edges 0,1,2,3 \
  --sequence-length 40
```

沿用 FR 的三档 selected variants：

| Slug | Raw | Kept | Rejected | Reject Ratio | Checked Nonzero | Top Violated Edge |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `k30_cov0p5_chk1` | 140 | 74 | 66 | 0.4714 | 109 | `d:22->d:13` |
| `k30_cov0p5_chk2` | 140 | 111 | 29 | 0.2071 | 109 | `d:22->d:13` |
| `k30_cov0p5_chk3` | 140 | 122 | 18 | 0.1286 | 109 | `d:22->d:13` |

## 3. SmartGen 原生 Transformer Autoencoder 结果

未过滤 baseline：

```bash
/home/heyang/miniconda3/envs/smartguard_env/bin/python -m causal_smart_home.cli smartgen-anomaly-eval \
  --synthetic-pkl /home/heyang/projects/SmartGen/anomaly_detection_pipeline/synthetic_data/sp_spring_generation_SPPC_th=0.915_gpt-4o_seq_filter_true.pkl \
  --out-dir outputs/sp_winter_to_spring_device_h0/smartgen_anomaly_eval_gpu \
  --dataset sp \
  --env spring \
  --tag unfiltered_filter_true \
  --device cuda \
  --cuda-visible-devices 0
```

Causal-filtered variants with common validation calibration：

```bash
/home/heyang/miniconda3/envs/smartguard_env/bin/python -m causal_smart_home.cli smartgen-anomaly-sweep-eval \
  --sweep-summary outputs/sp_winter_to_spring_device_h0/filter_sweep/filter_sweep_summary.csv \
  --out-dir outputs/sp_winter_to_spring_device_h0/smartgen_anomaly_sweep_eval_gpu_common_vld \
  --dataset sp \
  --env spring \
  --select-slugs k30_cov0p5_chk1,k30_cov0p5_chk2,k30_cov0p5_chk3 \
  --device cuda \
  --cuda-visible-devices 0 \
  --validation-pkl outputs/sp_winter_to_spring_device_h0/smartgen_anomaly_eval_gpu/unfiltered_filter_true_vld.pkl
```

结果表：

| Method | Synthetic Size | Train | Validation | Threshold | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `unfiltered_filter_true` | 140 | 112 | 28 | 0.0053 | 1.0000 | 0.8585 | 0.9239 |
| `k30_cov0p5_chk1` | 74 | 59 | 15 | 6.5453 | 0.0989 | 0.3478 | 0.1540 |
| `k30_cov0p5_chk2` | 111 | 88 | 23 | 5.7905 | 0.7486 | 0.7730 | 0.7606 |
| `k30_cov0p5_chk3` | 122 | 97 | 25 | 8.4056 | 0.7885 | 0.7788 | 0.7836 |

## 4. 结论

SP winter -> spring 没有复现 FR winter -> spring 的 F1 提升。这里未过滤 SmartGen `filter_true` baseline 已经很强，F1 为 0.9239；三档 causal-filtered variants 在 common validation calibration 下均低于 baseline，尤其 `chk1` 过滤过强后 recall 明显下降。

当前更稳妥的表述是：

```text
FR winter -> spring shows a positive case for causal filtering with common validation calibration,
while SP winter -> spring is a negative/diagnostic case where the same selected filter family
over-prunes useful synthetic sequences and underperforms the unfiltered SmartGen baseline.
```

因此，后续报告不应把 FR 的提升外推为普遍结论；更合理的下一步是继续扩展 US、night/multiple，或者在 SP 上探索 softer weighting / less aggressive selection，而不是只用 hard deletion。
