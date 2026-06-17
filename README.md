# CausalSmartHome

`CausalSmartHome` 是一个面向智能家居行为漂移实验的非侵入式因果胶水层项目。它把 SmartGen 风格的合成行为数据、GCAD-style 的 Granger 因果先验，以及 SmartGen/SmartGuard 下游异常检测流程连接起来，但不修改原始项目主体代码。

当前主线定位已经收敛为：把 GCAD-style causal prior 加到 SmartGen 的生成数据质量控制里，再沿用 SmartGen 自己的 Transformer Autoencoder 异常检测流程评估。SmartGuard wrapper 保留为辅助对照，用来观察同一批合成数据迁移到另一个异常检测框架时会发生什么。

本项目关注的问题是：

```text
能否从历史正常行为中挖掘因果结构，并用它约束或过滤合成智能家居行为数据，
从而影响后续异常检测模型的再训练效果？
```

## 项目定位

`CausalSmartHome` 不是 SmartGen、SmartGuard 或 GCAD 的替代实现，也不是把三个项目源码合并后的魔改版本。它的设计原则是：

```text
原项目主体代码保持不变
只在输入、输出和实验脚本边界做 adapter / wrapper / filter
因果先验作为额外提示或过滤信号接入生成数据流程
```

旧的失败/废弃项目 `CausalGenGuard` 不属于当前方案。

## 方法概览

SmartGen/SmartGuard 的行为序列通常是扁平化数值格式：

```text
[day, hour_slot, device_id, action_id, day, hour_slot, device_id, action_id, ...]
```

本项目先把离散行为事件转成多变量事件张量，再训练一个轻量 GCAD-style 梯度因果挖掘器，得到 action/device/device-action 级因果先验。该先验目前主要用于两个位置：

1. `causal_prompt.py`：生成 SmartGen prompt 中的 causal hints。
2. `causal_filter.py`：对 SmartGen 生成序列做因果一致性评分和过滤。

当前实验主要验证的是生成后过滤路径。

```text
SmartGen / SmartGuard numeric sequences
        |
        v
BehaviorSequence schema
        |
        v
Event Tensor Bridge
        |
        v
GCAD-style gradient causal prior
        |
        +--------------------------+
        |                          |
        v                          v
causal prompt hints          causal consistency filter
        |                          |
        v                          v
SmartGen generation      filtered synthetic sequences
        |                          |
        +-------------> downstream anomaly detection
```

## 目录结构

```text
CausalSmartHome/
  causal_smart_home/
    schema.py             # 行为事件和行为序列数据模型
    event_tensor.py       # 离散行为序列 -> [T, C] 事件张量
    causal_prior.py       # 轻量 GCAD-style 梯度因果挖掘器
    gcad_adapter.py       # 本地因果挖掘包装入口
    causal_prompt.py      # SmartGen causal hints 构造
    causal_filter.py      # 因果一致性评分和过滤
    smartgen_adapter.py   # SmartGen pkl 数据约定和转移提示
    smartgen_experiment.py # SmartGen Transformer Autoencoder 评估包装
    smartguard_adapter.py # SmartGuard 风格脚本 subprocess 包装
    pipeline.py           # 高层流程编排
    demo_data.py          # toy 数据
    cli.py                # 命令行入口
  docs/
    figures/framework.svg
    task*.md
  external_sources/
    SmartGuard -> /home/heyang/projects/SmartGuard
    SmartGen   -> /home/heyang/projects/SmartGen
    GCAD       -> /home/heyang/projects/GCAD
  external_sources_snapshot_from_tar/
  outputs/
  tests/
  pyproject.toml
  requirements.txt
```

`external_sources/` 推荐使用指向完整上游项目的本地软链接。`external_sources_snapshot_from_tar/` 只保留作为原 tar 包快照参考。

## 环境

本机推荐环境：

```bash
conda activate /home/heyang/miniconda3/envs/smartguard_env
cd /home/heyang/projects/CausalSmartHome
```

已验证环境：

```text
Python: /home/heyang/miniconda3/envs/smartguard_env/bin/python
Python version: 3.8.20
torch: 2.2.0
```

安装或补齐依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .[dev]
```

如果不做 editable install，请在项目根目录使用 `PYTHONPATH=.` 运行命令。

## GPU / CUDA 优先

SmartGen anomaly 相关 CLI 默认优先使用服务器第 0 张 NVIDIA GPU：

```text
--device cuda
--cuda-visible-devices 0
```

这里 GPU 是硬件，CUDA 是 PyTorch 调用 NVIDIA GPU 的计算后端。也就是说，本项目正式实验默认策略是“用 CUDA 调用第 0 张 GPU”。如果只想自动探测，可传 `--device auto`；如果临时排查或没有 GPU，可显式传 `--device cpu`。

在受限 shell 或沙箱里，`nvidia-smi` 可能看不到 `/dev/nvidia*`；在真实服务器终端中应能看到：

```text
NVIDIA GeForce RTX 3090
```

## external_sources 设置

检查软链接：

```bash
cd /home/heyang/projects/CausalSmartHome

readlink -f external_sources/SmartGuard
readlink -f external_sources/SmartGen
readlink -f external_sources/GCAD
```

期望输出：

```text
/home/heyang/projects/SmartGuard
/home/heyang/projects/SmartGen
/home/heyang/projects/GCAD
```

`build-prior`、`prompt`、`filter` 本身主要依赖显式传入的 `.pkl` 和 `.json` 路径，不强依赖 `external_sources/`。软链接主要用于后续调用 SmartGen/SmartGuard 原始实验流程时保持路径稳定。

## CLI 用法

查看帮助：

```bash
PYTHONPATH=. python -m causal_smart_home.cli --help
```

运行 toy demo：

```bash
PYTHONPATH=. python -m causal_smart_home.cli demo \
  --out-dir outputs/demo \
  --num-sequences 30 \
  --epochs 2 \
  --lag 3
```

从正常训练序列构建因果先验：

```bash
PYTHONPATH=. python -m causal_smart_home.cli build-prior \
  --train-pkl /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/winter/split_trn.pkl \
  --out-dir outputs/fr_winter_to_spring_device_h5e-05 \
  --lag 4 \
  --epochs 40 \
  --level device \
  --sparse-threshold 0.00005
```

US winter train 较大时，可以调大 batch size 提升 GPU prior 训练吞吐：

```bash
PYTHONPATH=. python -m causal_smart_home.cli build-prior \
  --train-pkl /home/heyang/projects/SmartGen/SmartGen/IoT_data/us/winter/split_trn.pkl \
  --out-dir outputs/us_winter_to_spring_device_h5e-05 \
  --lag 4 \
  --epochs 40 \
  --level device \
  --sparse-threshold 0.00005 \
  --batch-size 4096
```

构造带因果提示的 SmartGen prompt：

```bash
PYTHONPATH=. python -m causal_smart_home.cli prompt \
  --prior-json outputs/fr_winter_to_spring_device_h5e-05/causal_prior.json \
  --compressed-pkl path/to/smartgen_compressed.pkl \
  --device-info-json examples/device_info_toy.json \
  --original-context winter \
  --new-context spring \
  --out-prompt outputs/fr_winter_to_spring_device_h5e-05/causal_prompt.txt
```

过滤 SmartGen 原始生成序列：

```bash
PYTHONPATH=. python -m causal_smart_home.cli filter \
  --prior-json outputs/fr_winter_to_spring_device_h5e-05/causal_prior.json \
  --generated-pkl /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/spring/fr_spring_generation_SPPC_th=0.918_gpt-4o_seq.pkl \
  --out-pkl outputs/fr_winter_to_spring_device_h5e-05/fr_spring_generation_causal_kept.pkl \
  --out-scores outputs/fr_winter_to_spring_device_h5e-05/fr_spring_generation_causal_scores.json \
  --min-coverage 0.5
```

过滤 SmartGen 原 `filter_true` 序列：

```bash
PYTHONPATH=. python -m causal_smart_home.cli filter \
  --prior-json outputs/fr_winter_to_spring_device_h5e-05/causal_prior.json \
  --generated-pkl /home/heyang/projects/SmartGen/anomaly_detection_pipeline/synthetic_data/fr_spring_generation_SPPC_th=0.918_gpt-4o_seq_filter_true.pkl \
  --out-pkl outputs/fr_winter_to_spring_device_h5e-05/fr_spring_SPPC_filter_true_causal_kept.pkl \
  --out-scores outputs/fr_winter_to_spring_device_h5e-05/fr_spring_SPPC_filter_true_causal_scores.json \
  --min-coverage 0.5
```

扫描 causal filter 参数：

```bash
PYTHONPATH=. python -m causal_smart_home.cli sweep-filter \
  --prior-json outputs/fr_winter_to_spring_device_h5e-05/causal_prior.json \
  --generated-pkl /home/heyang/projects/SmartGen/anomaly_detection_pipeline/synthetic_data/fr_spring_generation_SPPC_th=0.918_gpt-4o_seq_filter_true.pkl \
  --out-dir outputs/fr_winter_to_spring_device_h5e-05/filter_sweep \
  --tag fr_spring_filter_true \
  --top-k-edges 10,20,30 \
  --min-coverages 0.3,0.5,0.7 \
  --min-checked-edges 0,1,2,3 \
  --sequence-length 40
```

输出文件：

```text
filter_sweep_summary.csv
filter_sweep_summary.json
fr_spring_filter_true_k30_cov0p5_chk2_kept.pkl
...
```

`min_checked_edges` 用来降低误删：当一条序列实际命中的因果边少于该值时，即使 coverage 低，也暂时保留，避免“证据不足时硬删除”。

`sequence_length 40` 会把写出的 kept pkl 统一 pad/truncate 到 SmartGuard 训练数据的长度；如果只是分析 SmartGen 原始变长序列，可以省略这个参数。

沿用 SmartGen 原 Transformer Autoencoder 异常检测流程评估一份合成数据：

```bash
PYTHONPATH=. python -m causal_smart_home.cli smartgen-anomaly-eval \
  --synthetic-pkl /home/heyang/projects/SmartGen/anomaly_detection_pipeline/synthetic_data/fr_spring_generation_SPPC_th=0.918_gpt-4o_seq_filter_true.pkl \
  --out-dir outputs/fr_winter_to_spring_device_h5e-05/smartgen_anomaly_eval \
  --dataset fr \
  --env spring \
  --tag unfiltered_filter_true \
  --device cuda \
  --cuda-visible-devices 0
```

批量把 causal filter sweep 产物送回 SmartGen 异常检测流程：

```bash
PYTHONPATH=. python -m causal_smart_home.cli smartgen-anomaly-sweep-eval \
  --sweep-summary outputs/fr_winter_to_spring_device_h5e-05/filter_sweep/filter_sweep_summary.csv \
  --out-dir outputs/fr_winter_to_spring_device_h5e-05/smartgen_anomaly_sweep_eval \
  --dataset fr \
  --env spring \
  --select-slugs k30_cov0p5_chk1,k30_cov0p5_chk2,k30_cov0p5_chk3 \
  --device cuda \
  --cuda-visible-devices 0
```

如需先检查路径、样本数和 train/validation 分割，可以加 `--dry-run`。CausalSmartHome 的 wrapper 复用 SmartGen 的 `TransformerAutoencoder` 和 Dataset 定义，但训练/阈值/评估循环是 CPU/GPU 自适应的；服务器资源说明要求只用第一张卡时，传入 `--cuda-visible-devices 0 --device cuda`。

如果要让多个模型使用同一份 validation 数据做阈值校准，可以加 `--validation-pkl`：

```bash
PYTHONPATH=. python -m causal_smart_home.cli smartgen-anomaly-sweep-eval \
  --sweep-summary outputs/fr_winter_to_spring_device_h5e-05/filter_sweep/filter_sweep_summary.csv \
  --out-dir outputs/fr_winter_to_spring_device_h5e-05/smartgen_anomaly_sweep_eval_gpu_common_vld \
  --dataset fr \
  --env spring \
  --select-slugs k30_cov0p5_chk1,k30_cov0p5_chk2,k30_cov0p5_chk3 \
  --device cuda \
  --cuda-visible-devices 0 \
  --validation-pkl outputs/fr_winter_to_spring_device_h5e-05/smartgen_anomaly_eval_gpu/unfiltered_filter_true_vld.pkl
```

如果要做 soft weighting 而不是 hard deletion，可以在 SmartGen anomaly eval 中传入 causal prior。该模式保留全部 synthetic sequences，只用 causal coverage 调整训练 loss 权重：

```bash
PYTHONPATH=. python -m causal_smart_home.cli smartgen-anomaly-eval \
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

批量调用 SmartGuard 原流程训练和评估作为辅助对照：

```bash
PYTHONPATH=. python -m causal_smart_home.cli smartguard-sweep-eval \
  --sweep-summary outputs/fr_winter_to_spring_device_h5e-05/filter_sweep/filter_sweep_summary.csv \
  --out-dir outputs/fr_winter_to_spring_device_h5e-05/smartguard_sweep_eval \
  --dataset fr \
  --select-slugs k30_cov0p5_chk1,k30_cov0p5_chk2,k30_cov0p5_chk3 \
  --epochs 60 \
  --threshold-percentage 95 \
  --sequence-length 40
```

这个命令不会改 SmartGuard 源码。它会在 `out-dir` 中为每个 selected slug 生成：

```text
*_merged_train.pkl
*_smartguard.pth
*_smartguard_eval.json
smartguard_sweep_eval_summary.csv
smartguard_sweep_eval_summary.json
```

如需先检查合并训练集和输出路径，可以加 `--dry-run`，不会真正训练模型。

## 当前实验快照

主实验设置：

```text
dataset: fr
source context: winter
target context: spring
causal level: device
lag: 4
epochs: 40
sparse threshold: 5e-05
```

主输出目录：

```text
outputs/fr_winter_to_spring_device_h5e-05/
```

对 raw SmartGen generated sequence 的过滤结果：

```text
raw generated: 137
kept: 119
rejected: 18
reject ratio: 13.14%
```

早期 SmartGen anomaly 快照：

| Method | Recall | Precision | F1 |
| --- | ---: | ---: | ---: |
| Original SPPC | 0.9886 | 0.7311 | 0.8406 |
| SPPC_CausalGCAD | 1.0000 | 0.6822 | 0.8111 |

这组早期结果说明：causal filter 会改变 Recall/Precision 权衡，但当时的 hard deletion 版本没有超过原 SPPC F1。因此它不能单独支撑“整体异常检测性能已经提升”的结论。

当前主线修正后，重点改为比较 SmartGen 原生 Transformer Autoencoder 训练口径下的：

```text
base / unfiltered filter_true / causal-filtered filter_true
```

新增 `smartgen-anomaly-eval` 与 `smartgen-anomaly-sweep-eval` 后，已在 GPU 上完成 SmartGen Transformer Autoencoder 正式评估：

未过滤 SmartGen `filter_true` 基线：

| Method | Synthetic Size | Threshold | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| unfiltered_filter_true | 125 | 1.3921 | 0.9886 | 0.7311 | 0.8406 |

使用未过滤 `filter_true` 的 validation split 作为公共阈值校准集后，三档 causal filter 结果为：

| Method | Synthetic Size | Threshold | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| unfiltered_filter_true | 125 | 1.3921 | 0.9886 | 0.7311 | 0.8406 |
| k30_cov0p5_chk1 | 109 | 1.6313 | 0.9886 | 0.7768 | 0.8700 |
| k30_cov0p5_chk2 | 118 | 0.0136 | 1.0000 | 0.6822 | 0.8111 |
| k30_cov0p5_chk3 | 121 | 0.0066 | 1.0000 | 0.6822 | 0.8111 |

当前最有价值的结论是：causal filter 本身有潜力改善 SmartGen 生成数据，最佳 `k30_cov0p5_chk1` 在公共 validation 校准后 F1 从未过滤的 0.8406 提升到 0.8700；但 hard deletion 需要配套阈值校准，否则会因为 validation 分布过窄而牺牲 Precision。

论文或报告中目前最稳妥的表述是：CausalSmartHome 已经把 GCAD-style causal filter 接回 SmartGen 原异常检测实验路径；在 FR winter -> spring 的 GPU 复现实验中，causal filtering + common validation calibration 比未过滤 SmartGen `filter_true` 取得更高 F1。

FR 上也回测了 soft weighting。最佳 weighted 设置 `top_k=30, floor=0.2, power=1` 的 F1 为 0.8571，高于未过滤 baseline 的 0.8406，但低于 hard deletion 最佳 0.8700。详细记录见 `docs/task10_fr_weighted_smartgen.md`。

### SP winter -> spring SmartGen 扩展

按同一 SmartGen 原生异常检测口径补跑了 SP winter -> spring。SP 的 `sparse-threshold=5e-05` device prior 被 sparsify 后没有非零边，filter sweep 退化为全保留；因此 SP 主实验使用 `sparse-threshold=0` 的 device prior。

```text
dataset: sp
source context: winter
target context: spring
causal level: device
lag: 4
epochs: 40
sparse threshold: 0
```

主输出目录：

```text
outputs/sp_winter_to_spring_device_h0/
```

三档 causal filter 的过滤规模：

| Filter | Raw | Kept | Rejected | Reject Ratio |
| --- | ---: | ---: | ---: | ---: |
| k30_cov0p5_chk1 | 140 | 74 | 66 | 47.14% |
| k30_cov0p5_chk2 | 140 | 111 | 29 | 20.71% |
| k30_cov0p5_chk3 | 140 | 122 | 18 | 12.86% |

使用未过滤 `filter_true` 的 validation split 作为公共阈值校准集后，SP 结果为：

| Method | Synthetic Size | Threshold | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| unfiltered_filter_true | 140 | 0.0053 | 1.0000 | 0.8585 | 0.9239 |
| k30_cov0p5_chk1 | 74 | 6.5453 | 0.0989 | 0.3478 | 0.1540 |
| k30_cov0p5_chk2 | 111 | 5.7905 | 0.7486 | 0.7730 | 0.7606 |
| k30_cov0p5_chk3 | 122 | 8.4056 | 0.7885 | 0.7788 | 0.7836 |

SP 结论与 FR 不同：未过滤 SmartGen baseline 已经较强，三档 hard-deletion causal filter 均未超过 baseline。详细记录见 `docs/task7_sp_smartgen_anomaly.md`。

随后补做了 SP 被删样本分布诊断和温和过滤复测。被删样本通常比 kept 样本更远离 spring normal test，但即使只删除 6-15 条样本，filtered model 对 baseline validation 的重构 loss 也会从 0.0007 升到 1.9021-9.9090，最佳温和变体 F1 只有 0.8118。因此 SP 的主要问题不是简单参数过狠，而是 hard deletion 会破坏 SmartGen synthetic training distribution。详细记录见 `docs/task8_sp_filter_diagnostics.md`。

再进一步实现了 causal soft weighting：保留全部 140 条 synthetic sequences，用 `weight = floor + (1 - floor) * causal_coverage ** power` 给训练 loss 加权。SP 最佳 weighted 结果为 `top_k=30, floor=0.2, power=1`，F1 达到 0.9189，明显好于 hard deletion，但仍略低于未过滤 baseline 的 0.9239。详细记录见 `docs/task9_sp_weighted_smartgen.md`。

### US winter -> spring SmartGen 扩展

US winter -> spring 已按同一流程完成。US baseline 本身很强，hard deletion 和 mild hard deletion 均未超过 baseline；soft weighting 保持了 baseline 水平但没有额外提升。

| Method | Synthetic Size | Threshold | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| unfiltered_filter_true | 94 | 0.1698 | 1.0000 | 0.8697 | 0.9303 |
| best hard deletion (`k30_cov0p5_chk2`) | 55 | 0.4494 | 1.0000 | 0.8595 | 0.9244 |
| best mild hard deletion (`k30_cov0p3_chk3`) | 74 | 0.2161 | 1.0000 | 0.8593 | 0.9243 |
| soft weighting (`k30_floor0p2`) | 94 | 0.1678 | 1.0000 | 0.8697 | 0.9303 |

三数据集当前结论是：FR 是 hard deletion 正例；SP/US 的 baseline 已较强，hard deletion 不稳，soft weighting 更安全但不带来额外提升。详细记录见 `docs/task11_us_smartgen_anomaly.md`。

### SmartGuard wrapper 对照评估（辅助）

随后使用 `smartguard-sweep-eval` 在同一 SmartGuard wrapper 口径下补跑了 `base_only`、未过滤 `filter_true` 和三档 causal filter 对照。

```text
dataset: fr
epochs: 60
threshold percentage: 95
attacks: SD, MD, DM, DD
```

| Method | Added Synthetic | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: |
| base_only | 0 | 0.9935 | 0.9766 | 0.9850 |
| unfiltered_filter_true | 125 | 0.9425 | 0.9731 | 0.9575 |
| k30_cov0p5_chk1 | 109 | 0.9622 | 0.9736 | 0.9679 |
| k30_cov0p5_chk2 | 118 | 0.9533 | 0.9682 | 0.9607 |
| k30_cov0p5_chk3 | 121 | 0.9522 | 0.9793 | 0.9655 |

同一 wrapper 口径下，causal filter 相比未过滤合成数据有所改善，但仍未超过 `base_only`。详细记录见 `docs/task5_smartguard_sweep_eval.md`。

## 测试

在项目根目录运行：

```bash
PYTHONPATH=. pytest -q
```

也可以显式使用推荐环境中的 Python：

```bash
/home/heyang/miniconda3/envs/smartguard_env/bin/python -m pytest -q
```

当前验证结果：

```text
PYTHONPATH=. pytest -q
8 passed, 3 skipped

/home/heyang/miniconda3/envs/smartguard_env/bin/python -m pytest -q
11 passed, 1 warning
```

如果某个环境没有安装 `torch`，依赖因果训练的测试会自动跳过，非 torch 测试仍可运行。

## 已知限制

- `causal_prior.py` 是轻量 GCAD-style 实现，迁移了 GCAD 的梯度因果思想，但不是直接运行原 GCAD 实验脚本。
- 当前证据显示 FR winter -> spring 是 hard deletion 正例，SP/US winter -> spring 是 baseline 强、causal post-filter 不提升的边界案例；仍需扩展到 night/multiple 才能说明普遍性。
- SmartGen 原异常检测脚本直接调用 `.cuda()`；CausalSmartHome wrapper 已改为 CPU/GPU 自适应，但 CPU 跑完整 sweep 会比 GPU 慢。
- hard deletion 可能让训练/验证分布变窄，导致 anomaly threshold 过低并增加误报。
- 后续应重点尝试阈值校准、软权重过滤、多数据集验证，以及被过滤样本的可解释分析。

## Git 管理

`.gitignore` 已忽略本地大文件和实验产物，包括：

```text
external_sources/*
external_sources_snapshot_from_tar/
outputs/
*.pkl
*.pt
*.pth
*.csv
*.tar.gz
```

建议提交源码、测试、文档和小型示例。上游项目、数据集、生成序列、模型权重和实验输出默认保留在本地。
