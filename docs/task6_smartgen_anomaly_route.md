# 任务六：回到 SmartGen 原生异常检测路线

## 1. 为什么主线改回 SmartGen

SmartGen 本身已经有完整的 context drift anomaly detection 实验链路：用某个源上下文的正常行为生成目标上下文合成数据，再用这些合成数据训练 Transformer Autoencoder，最后在目标上下文 normal / attack 数据上评估异常检测效果。

因此，CausalSmartHome 里 GCAD-style causal prior 最自然的位置不是替换下游异常检测模型，而是改进 SmartGen 的生成数据质量控制：

```text
SmartGen generated sequences
        |
        v
GCAD-style causal consistency filter
        |
        v
SmartGen Transformer Autoencoder anomaly detection
```

SmartGuard wrapper 仍然有价值，但它更适合作为横向对照：同一批合成数据如果送到另一个异常检测框架中，是否也能改善未过滤版本。它不是当前主贡献的唯一评估口径。

## 2. 新增实现

新增模块：

```text
causal_smart_home/smartgen_experiment.py
```

新增 CLI：

```text
smartgen-anomaly-eval
smartgen-anomaly-sweep-eval
```

它们做的事情是：

1. 读取 SmartGen 或 CausalSmartHome 产出的 synthetic pkl。
2. 按 SmartGen 原逻辑切分 train / validation。
3. 调用 SmartGen 的 `Anomaly_Detection_pipeline_model.py` 中的 `train`、`find_threshold`、`evaluate`。
4. 把模型、分割数据和结果写到 CausalSmartHome 的 `outputs/`，不写回 SmartGen 源目录。

对 `spring` 和 `night`，默认随机切成 80% train、20% validation；对 `multiple`，沿用 SmartGen 原逻辑，把同一份 synthetic data 同时作为 train 和 validation。

## 3. 已验证的 dry-run

当前机器没有 CUDA：

```text
torch.cuda.is_available(): False
```

SmartGen 原异常检测代码直接调用 `.cuda()`，所以正式训练不能在当前 CPU 环境完成。已完成 dry-run，确认路径、样本数和输出文件没有问题。

单个未过滤 SmartGen `filter_true`：

```text
synthetic size: 125
train size: 100
validation size: 25
```

三档 causal filter sweep：

| Slug | Synthetic Size | Train Size | Validation Size |
| --- | ---: | ---: | ---: |
| k30_cov0p5_chk1 | 109 | 87 | 22 |
| k30_cov0p5_chk2 | 118 | 94 | 24 |
| k30_cov0p5_chk3 | 121 | 96 | 25 |

dry-run 输出目录：

```text
outputs/fr_winter_to_spring_device_h5e-05/smartgen_anomaly_dryrun/
outputs/fr_winter_to_spring_device_h5e-05/smartgen_anomaly_dryrun_sweep/
```

这些目录在 `outputs/` 下，默认不进入 Git。

## 4. GPU 环境下的正式命令

未过滤 SmartGen `filter_true`：

```bash
PYTHONPATH=. python -m causal_smart_home.cli smartgen-anomaly-eval \
  --synthetic-pkl /home/heyang/projects/SmartGen/anomaly_detection_pipeline/synthetic_data/fr_spring_generation_SPPC_th=0.918_gpt-4o_seq_filter_true.pkl \
  --out-dir outputs/fr_winter_to_spring_device_h5e-05/smartgen_anomaly_eval \
  --dataset fr \
  --env spring \
  --tag unfiltered_filter_true
```

causal filter sweep：

```bash
PYTHONPATH=. python -m causal_smart_home.cli smartgen-anomaly-sweep-eval \
  --sweep-summary outputs/fr_winter_to_spring_device_h5e-05/filter_sweep/filter_sweep_summary.csv \
  --out-dir outputs/fr_winter_to_spring_device_h5e-05/smartgen_anomaly_sweep_eval \
  --dataset fr \
  --env spring \
  --select-slugs k30_cov0p5_chk1,k30_cov0p5_chk2,k30_cov0p5_chk3
```

建议先跑 FR winter -> spring，再扩展到 SP/US 和 night/multiple。只有在 SmartGen 原生异常检测口径下比较完 `unfiltered_filter_true` 与 causal-filtered variants，才能严谨地说 causal filter 是否真的改进了 SmartGen。

## 5. 如何解释现有 SmartGuard 结果

SmartGuard wrapper 当前结果是：

| Method | Added Synthetic | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: |
| base_only | 0 | 0.9935 | 0.9766 | 0.9850 |
| unfiltered_filter_true | 125 | 0.9425 | 0.9731 | 0.9575 |
| k30_cov0p5_chk1 | 109 | 0.9622 | 0.9736 | 0.9679 |
| k30_cov0p5_chk2 | 118 | 0.9533 | 0.9682 | 0.9607 |
| k30_cov0p5_chk3 | 121 | 0.9522 | 0.9793 | 0.9655 |

它说明 causal filter 相比未过滤合成数据更好，但没有超过 SmartGuard 的 base-only。这个结果不否定项目可行性，因为 SmartGuard 不是 SmartGen 的原实验口径，而且 base-only 本身已经很强。更合理的写法是：

```text
Under a SmartGuard auxiliary evaluation, causal-filtered synthetic data improves over unfiltered synthetic data, while base-only remains strongest.
```

主结论应等待 SmartGen Transformer Autoencoder 路线的正式 GPU 结果。
