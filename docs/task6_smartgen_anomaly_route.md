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
3. 复用 SmartGen `models1.py` 中的 `TransformerAutoencoder` 和 Dataset 定义。
4. 在 CausalSmartHome 中执行 CPU/GPU 自适应的 train / find threshold / evaluate 循环。
5. 支持用 `--validation-pkl` 指定公共 validation set 做阈值校准。
6. 把模型、分割数据和结果写到 CausalSmartHome 的 `outputs/`，不写回 SmartGen 源目录。

对 `spring` 和 `night`，默认随机切成 80% train、20% validation；对 `multiple`，沿用 SmartGen 原逻辑，把同一份 synthetic data 同时作为 train 和 validation。

## 3. 正式 SmartGen anomaly 结果

当前机器没有 CUDA：

```text
torch.cuda.is_available(): False
```

CausalSmartHome wrapper 已绕开 SmartGen 原脚本里的硬编码 `.cuda()`，在 CPU 上跑完了 FR winter -> spring 的 SmartGen Transformer Autoencoder 评估。

### 默认各自 validation

每个方法用自己的 synthetic split 训练并校准阈值：

| Method | Synthetic Size | Train | Validation | Threshold | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unfiltered_filter_true | 125 | 100 | 25 | 1.7035 | 0.9773 | 0.7478 | 0.8473 |
| k30_cov0p5_chk1 | 109 | 87 | 22 | 0.0004 | 1.0000 | 0.6331 | 0.7753 |
| k30_cov0p5_chk2 | 118 | 94 | 24 | 0.0005 | 1.0000 | 0.6667 | 0.8000 |
| k30_cov0p5_chk3 | 121 | 96 | 25 | 0.0003 | 1.0000 | 0.6331 | 0.7753 |

这组结果不是简单说明 causal filter 失败，而是暴露出阈值校准问题：filtered synthetic validation 太容易被模型重构，validation loss 接近 0，导致 anomaly threshold 极低，测试时 Precision 被误报拖低。

输出目录：

```text
outputs/fr_winter_to_spring_device_h5e-05/smartgen_anomaly_eval_cpu/
outputs/fr_winter_to_spring_device_h5e-05/smartgen_anomaly_sweep_eval_cpu/
```

### 公共 validation 校准

用未过滤 `filter_true` 的 validation split 作为公共阈值校准集后：

| Method | Synthetic Size | Threshold | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| unfiltered_filter_true | 125 | 1.7035 | 0.9773 | 0.7478 | 0.8473 |
| k30_cov0p5_chk1 | 109 | 1.6970 | 0.9773 | 0.7679 | 0.8600 |
| k30_cov0p5_chk2 | 118 | 0.0098 | 1.0000 | 0.6822 | 0.8111 |
| k30_cov0p5_chk3 | 121 | 0.0055 | 1.0000 | 0.6822 | 0.8111 |

这里 `k30_cov0p5_chk1` 超过了未过滤版本，说明 causal filter 的收益可能被阈值校准方式遮住。当前最准确的结论是：

```text
Causal filtering can improve SmartGen synthetic data under the native Transformer Autoencoder evaluation when paired with a less brittle validation calibration.
```

## 4. 正式命令

未过滤 SmartGen `filter_true`：

```bash
PYTHONPATH=. python -m causal_smart_home.cli smartgen-anomaly-eval \
  --synthetic-pkl /home/heyang/projects/SmartGen/anomaly_detection_pipeline/synthetic_data/fr_spring_generation_SPPC_th=0.918_gpt-4o_seq_filter_true.pkl \
  --out-dir outputs/fr_winter_to_spring_device_h5e-05/smartgen_anomaly_eval \
  --dataset fr \
  --env spring \
  --tag unfiltered_filter_true \
  --device auto
```

causal filter sweep：

```bash
PYTHONPATH=. python -m causal_smart_home.cli smartgen-anomaly-sweep-eval \
  --sweep-summary outputs/fr_winter_to_spring_device_h5e-05/filter_sweep/filter_sweep_summary.csv \
  --out-dir outputs/fr_winter_to_spring_device_h5e-05/smartgen_anomaly_sweep_eval \
  --dataset fr \
  --env spring \
  --select-slugs k30_cov0p5_chk1,k30_cov0p5_chk2,k30_cov0p5_chk3 \
  --device auto
```

公共 validation 校准：

```bash
PYTHONPATH=. python -m causal_smart_home.cli smartgen-anomaly-sweep-eval \
  --sweep-summary outputs/fr_winter_to_spring_device_h5e-05/filter_sweep/filter_sweep_summary.csv \
  --out-dir outputs/fr_winter_to_spring_device_h5e-05/smartgen_anomaly_sweep_eval_cpu_common_vld \
  --dataset fr \
  --env spring \
  --select-slugs k30_cov0p5_chk1,k30_cov0p5_chk2,k30_cov0p5_chk3 \
  --device cpu \
  --validation-pkl outputs/fr_winter_to_spring_device_h5e-05/smartgen_anomaly_eval_cpu/unfiltered_filter_true_vld.pkl
```

建议接下来扩展到 SP/US 和 night/multiple，确认这个提升不是 FR spring 的偶然现象。

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

主结论应优先使用 SmartGen Transformer Autoencoder 路线；SmartGuard 结果作为辅助稳健性分析。
