# 任务十一：US winter -> spring SmartGen 原生异常检测

本轮把 FR/SP 之后的第三个数据集 US winter -> spring 跑完，用于判断 causal filter / soft weighting 的可迁移性。

流程：

```text
build prior -> causal filter sweep -> GPU unfiltered baseline
-> baseline validation common calibration
-> GPU hard-deletion variants
-> GPU soft-weighted variants
-> summarize F1 / Recall / Precision / threshold
```

所有正式 SmartGen 训练都使用：

```bash
--device cuda --cuda-visible-devices 0
```

SmartGen / SmartGuard 源码未改动。

## 1. 数据和 prior

```text
dataset: us
source context: winter
target context: spring
synthetic: /home/heyang/projects/SmartGen/anomaly_detection_pipeline/synthetic_data/us_spring_generation_SPPC_th=0.905_gpt-4o_seq_filter_true.pkl
normal test: /home/heyang/projects/SmartGen/anomaly_detection_pipeline/test/us/spring/test.pkl
attack: /home/heyang/projects/SmartGen/anomaly_detection_pipeline/attack/us/labeled_us_spring_attack_heater.pkl
```

样本数：

| File | Count |
| --- | ---: |
| US winter `split_trn.pkl` | 30105 |
| US spring SmartGen `filter_true` synthetic | 94 |
| US spring normal test | 3016 |
| US spring heater attack | 3016 |

US winter train 明显大于 FR/SP，因此本轮给 `build-prior` 增加了向后兼容的 `--batch-size` 参数。默认仍为 64；US prior 使用 4096 以提升 GPU 吞吐。

```bash
/home/heyang/miniconda3/envs/smartguard_env/bin/python -m causal_smart_home.cli build-prior \
  --train-pkl /home/heyang/projects/SmartGen/SmartGen/IoT_data/us/winter/split_trn.pkl \
  --out-dir outputs/us_winter_to_spring_device_h5e-05 \
  --lag 4 \
  --epochs 40 \
  --level device \
  --sparse-threshold 0.00005 \
  --batch-size 4096
```

Prior 摘要：

| Item | Value |
| --- | ---: |
| Device channels | 23 |
| Non-zero matrix entries | 40 |
| Final train loss | 0.001207 |

## 2. Causal filter sweep

```bash
/home/heyang/miniconda3/envs/smartguard_env/bin/python -m causal_smart_home.cli sweep-filter \
  --prior-json outputs/us_winter_to_spring_device_h5e-05/causal_prior.json \
  --generated-pkl /home/heyang/projects/SmartGen/anomaly_detection_pipeline/synthetic_data/us_spring_generation_SPPC_th=0.905_gpt-4o_seq_filter_true.pkl \
  --out-dir outputs/us_winter_to_spring_device_h5e-05/filter_sweep \
  --tag us_spring_filter_true \
  --top-k-edges 10,20,30 \
  --min-coverages 0.3,0.5,0.7 \
  --min-checked-edges 0,1,2,3 \
  --sequence-length 40
```

FR-style selected variants：

| Slug | Raw | Kept | Rejected | Reject Ratio | Checked Nonzero | Top Violated Edge |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `k30_cov0p5_chk1` | 94 | 46 | 48 | 0.5106 | 83 | `d:27->d:15` |
| `k30_cov0p5_chk2` | 94 | 55 | 39 | 0.4149 | 83 | `d:27->d:15` |
| `k30_cov0p5_chk3` | 94 | 72 | 22 | 0.2340 | 83 | `d:27->d:15` |

Mild variants：

| Slug | Raw | Kept | Rejected | Reject Ratio |
| --- | ---: | ---: | ---: | ---: |
| `k10_cov0p3_chk2` | 94 | 82 | 12 | 0.1277 |
| `k20_cov0p3_chk2` | 94 | 75 | 19 | 0.2021 |
| `k30_cov0p3_chk3` | 94 | 74 | 20 | 0.2128 |

## 3. SmartGen anomaly 结果

未过滤 baseline：

```bash
/home/heyang/miniconda3/envs/smartguard_env/bin/python -m causal_smart_home.cli smartgen-anomaly-eval \
  --synthetic-pkl /home/heyang/projects/SmartGen/anomaly_detection_pipeline/synthetic_data/us_spring_generation_SPPC_th=0.905_gpt-4o_seq_filter_true.pkl \
  --out-dir outputs/us_winter_to_spring_device_h5e-05/smartgen_anomaly_eval_gpu \
  --dataset us \
  --env spring \
  --tag unfiltered_filter_true \
  --device cuda \
  --cuda-visible-devices 0
```

使用 baseline validation 做 common calibration 后，结果为：

| Group | Method | Synthetic Size | Rejected | Threshold | Val Loss Avg | Test Loss Avg | Recall | Precision | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | `unfiltered_filter_true` | 94 | 0 | 0.1698 | 0.0892 | 3.0165 | 1.0000 | 0.8697 | 0.9303 |
| Hard | `k30_cov0p5_chk1` | 46 | 48 | 2.0970 | 0.4604 | 4.2328 | 0.9433 | 0.7317 | 0.8242 |
| Hard | `k30_cov0p5_chk2` | 55 | 39 | 0.4494 | 0.2292 | 3.5105 | 1.0000 | 0.8595 | 0.9244 |
| Hard | `k30_cov0p5_chk3` | 72 | 22 | 0.0111 | 0.0035 | 3.1414 | 1.0000 | 0.8112 | 0.8958 |
| Mild | `k10_cov0p3_chk2` | 82 | 12 | 0.0030 | 0.0011 | 2.8128 | 1.0000 | 0.7063 | 0.8279 |
| Mild | `k20_cov0p3_chk2` | 75 | 19 | 0.3400 | 0.1203 | 3.7338 | 1.0000 | 0.8481 | 0.9178 |
| Mild | `k30_cov0p3_chk3` | 74 | 20 | 0.2161 | 0.1041 | 3.9405 | 1.0000 | 0.8593 | 0.9243 |
| Weighted | `weighted_k30_floor0p2_power1` | 94 | 0 | 0.1678 | 0.0881 | 2.8593 | 1.0000 | 0.8697 | 0.9303 |
| Weighted | `weighted_k30_floor0p5_power1` | 94 | 0 | 0.1868 | 0.0981 | 3.1235 | 1.0000 | 0.8697 | 0.9303 |
| Weighted | `weighted_k30_floor0p8_power1` | 94 | 0 | 0.1742 | 0.0915 | 3.1212 | 1.0000 | 0.8697 | 0.9303 |

## 4. 三数据集结论

| Dataset | Baseline F1 | Best Hard F1 | Best Soft Weighting F1 | Takeaway |
| --- | ---: | ---: | ---: | --- |
| FR | 0.8406 | 0.8700 | 0.8571 | hard deletion 是正例 |
| SP | 0.9239 | 0.8118 | 0.9189 | soft weighting 缓解 hard deletion，但不提升 |
| US | 0.9303 | 0.9244 | 0.9303 | baseline 已强，causal methods 不提升 |

## 5. 结论

US winter -> spring 更接近 SP，而不是 FR：未过滤 SmartGen baseline 已经很强，hard deletion 和 mild hard deletion 均未超过 baseline；soft weighting 保持了 baseline F1，但没有额外提升。

当前最稳妥的论文/报告表述应是：

```text
Causal filtering can improve SmartGen in a positive FR case, but the effect is not universal.
On SP and US, unfiltered SmartGen filter_true baselines are already strong; hard deletion can
hurt, while soft weighting is safer and usually preserves baseline-level performance.
```

下一步更像方法设计问题，而不是继续盲扫阈值：

1. 做 hybrid selector：根据 common validation compatibility 判断是否使用 hard deletion，否则回退 soft weighting。
2. 报告 FR/SP/US 三数据集，把 FR 作为正例，把 SP/US 作为稳定性边界。
3. 若继续优化，尝试把 causal score 用在 generation-stage prompt/reranking，而不是只在 post-generation training set 上删/降权。
