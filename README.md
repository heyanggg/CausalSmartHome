# CausalSmartHome

`CausalSmartHome` 是一个面向智能家居行为漂移实验的非侵入式因果胶水层项目。它把 SmartGen 风格的合成行为数据、GCAD-style 的 Granger 因果先验，以及 SmartGuard/SmartGen 下游异常检测流程连接起来，但不修改原始项目主体代码。

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

批量调用 SmartGuard 原流程训练和评估：

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

下游异常检测结果：

| Method | Recall | Precision | F1 |
| --- | ---: | ---: | ---: |
| Original SPPC | 0.9886 | 0.7311 | 0.8406 |
| SPPC_CausalGCAD | 1.0000 | 0.6822 | 0.8111 |

当前结论：

```text
因果过滤在当前 FR winter -> spring 实验中提升了 Recall，但降低了 Precision。
硬删除式 causal filter 尚未带来超过原 SPPC baseline 的 F1 提升。
```

因此论文或报告中应表述为“因果过滤改变了召回率和精确率的权衡”，不要直接写成“已经提升整体异常检测性能”。

### SmartGuard wrapper 对照评估

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
8 passed, 1 warning
```

如果某个环境没有安装 `torch`，依赖因果训练的测试会自动跳过，非 torch 测试仍可运行。

## 已知限制

- `causal_prior.py` 是轻量 GCAD-style 实现，迁移了 GCAD 的梯度因果思想，但不是直接运行原 GCAD 实验脚本。
- 当前证据主要证明工程链路跑通，还不能证明最终异常检测 F1 已经提升。
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
