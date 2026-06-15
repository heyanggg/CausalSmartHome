# 任务五：SmartGuard wrapper 正式评估记录

## 1. 目的

本轮实验使用 CausalSmartHome 新增的 `smartguard-eval` / `smartguard-sweep-eval` wrapper，在不修改 SmartGuard 源码和原始数据的前提下，验证不同 causal filter 输出对 SmartGuard 训练和异常检测结果的影响。

与早期 `outputs/fr_winter_to_spring_device_h5e-05/anomaly_detection_*.json` 快照不同，本轮结果全部来自同一套 SmartGuard wrapper 口径，因此内部可直接比较。

## 2. 实验设置

```text
dataset: fr
source context: winter
target context: spring
base train: /home/heyang/projects/SmartGuard/data/fr_data/fr_trn_instance_10.pkl
base train size: 2233
sequence length: 40
epochs: 60
threshold percentage: 95
attacks: SD, MD, DM, DD
```

评估输出目录：

```text
outputs/fr_winter_to_spring_device_h5e-05/smartguard_sweep_eval/
```

该目录位于 `outputs/` 下，包含 pkl、pth 和 json 实验产物，默认不提交到 Git。

## 3. 对照组

| Method | Added Synthetic | TP | TN | FP | FN | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base_only | 0 | 5337 | 2436 | 128 | 35 | 0.9935 | 0.9766 | 0.9850 |
| unfiltered_filter_true | 125 | 5063 | 2424 | 140 | 309 | 0.9425 | 0.9731 | 0.9575 |
| k30_cov0p5_chk1 | 109 | 5169 | 2424 | 140 | 203 | 0.9622 | 0.9736 | 0.9679 |
| k30_cov0p5_chk2 | 118 | 5121 | 2396 | 168 | 251 | 0.9533 | 0.9682 | 0.9607 |
| k30_cov0p5_chk3 | 121 | 5115 | 2456 | 108 | 257 | 0.9522 | 0.9793 | 0.9655 |

## 4. 结论

在同一 SmartGuard wrapper 口径下，`base_only` 表现最好，F1 为 0.9850。

加入未过滤的 SmartGen `filter_true` 合成数据后，Recall 从 0.9935 降到 0.9425，F1 从 0.9850 降到 0.9575。

三档 causal filter 都比未过滤合成数据好，其中 `k30_cov0p5_chk1` 最好，F1 为 0.9679；但三档 causal filter 仍未超过 `base_only`。

因此当前最稳妥的论文表述是：

```text
Causal filter can improve the quality of synthetic data relative to unfiltered SmartGen filter_true data under this SmartGuard wrapper evaluation, but synthetic augmentation still underperforms the base-only SmartGuard training setup on FR winter -> spring.
```

不要写成：

```text
CausalSmartHome already improves SmartGuard's final anomaly detection F1.
```

## 5. 后续方向

1. 优先检查合成数据与 SmartGuard 训练分布是否错位，尤其是 padding、时间槽、device/action 编码和序列长度。
2. 加入目标上下文 validation normal 数据重新校准 threshold，而不是固定使用原 SmartGuard validation set 的 95 分位。
3. 尝试 soft weighting / curriculum sampling，而不是把 kept synthetic data 直接硬拼到训练集。
4. 扩展到 SP/US 和其他 context shift，确认 FR winter -> spring 是否只是 base-only 特别强。
