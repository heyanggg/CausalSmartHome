# 任务四/五：胶水层实现与测试记录

## 1. 实现原则

本项目采用非侵入式胶水层实现。`external_sources/SmartGuard`、`external_sources/SmartGen`、`external_sources/GCAD` 只作为源码快照和适配对象保留，胶水层代码位于 `causal_smart_home/`，不改写三个原项目的模型主体和训练脚本。

核心实现思路：

1. 用 `schema.py` 统一 SmartGuard/SmartGen 的 `[day, hour, device, action]` 行为序列格式。
2. 用 `event_tensor.py` 将离散行为序列转为 GCAD 可处理的 `[T, C]` 多变量事件张量。
3. 用 `causal_prior.py` 实现轻量 GCAD-style 梯度因果挖掘器，训练普通预测器并对各输出通道损失反向传播，得到 action/device 级因果先验。
4. 用 `causal_prompt.py` 把因果先验转成 SmartGen prompt 中的 `causal_hints` JSON。
5. 用 `causal_filter.py` 在 SmartGen 生成后增加因果一致性过滤。
6. 用 `smartguard_adapter.py` 以 subprocess 方式调用未修改的 SmartGuard 训练/评估脚本。
7. 用 `smartgen_adapter.py` 复用 SmartGen 的 pkl 数据约定和转移 hints。
8. 用 `gcad_adapter.py` 封装 GCAD-style 因果挖掘；若需要复现原 GCAD，可准备其 `train.csv/test.csv` 数据目录并直接运行 `external_sources/GCAD`。

## 2. 代码结构

```text
CausalSmartHome/
  causal_smart_home/
    schema.py
    event_tensor.py
    causal_prior.py
    gcad_adapter.py
    causal_prompt.py
    causal_filter.py
    smartgen_adapter.py
    smartguard_adapter.py
    pipeline.py
    cli.py
    demo_data.py
  external_sources/
    GCAD/
    SmartGen/
    SmartGuard/
  docs/
    task1_source_review.md
    task2_feasibility_and_plan.md
    task3_chinese_paper_draft.md
    figures/framework.svg
  tests/
  examples/
  scripts/
```

## 3. 已执行测试

测试环境中没有包含完整 FR/SP/US 数据集，也无法在容器中直接拉取 GitHub 数据文件；因此，本次完成的是胶水层可执行性测试和 toy 数据端到端测试，而不是原论文级全量复现实验。

### 3.1 单元测试

命令：

```bash
cd /mnt/data/CausalSmartHome
PYTHONPATH=. pytest -q tests
```

结果：

```text
4 passed in 7.84s
```

覆盖内容：

- `test_event_tensor.py`：检查行为序列到事件张量的通道和形状。
- `test_causal_prior.py`：检查 GCAD-style 因果先验可训练、可序列化、可提取 top edges。
- `test_prompt_filter.py`：检查 prompt 中包含 causal hints，且生成序列可被因果一致性过滤器评分。
- `test_cli_demo.py`：检查命令行 demo 能生成 `causal_prior.json`、`causal_smartgen_prompt.txt`、`causal_filter_scores.json`。

### 3.2 Toy 端到端 demo

命令：

```bash
cd /mnt/data/CausalSmartHome
PYTHONPATH=. python -m causal_smart_home.cli demo --out-dir outputs/demo --num-sequences 30 --epochs 2 --lag 3
```

结果摘要：

```json
{
  "normal_sequences": 30,
  "candidate_sequences": 5,
  "kept_sequences": 4,
  "rejected_sequences": 1
}
```

生成文件：

- `outputs/demo/toy_normal.pkl`
- `outputs/demo/toy_generated_candidates.pkl`
- `outputs/demo/causal_prior.json`
- `outputs/demo/causal_smartgen_prompt.txt`
- `outputs/demo/causal_filter_scores.json`
- `outputs/demo/toy_generated_kept.pkl`
- `outputs/demo/demo_report.json`

## 4. 真实数据运行方式

真实实验需要把 SmartGen/SmartGuard 论文使用的 FR/SP/US 数据集放入项目或原项目约定目录。随后按以下流程运行：

```bash
csh build-prior \
  --train-pkl path/to/normal_train.pkl \
  --out-dir outputs/fr_st \
  --lag 4 \
  --epochs 80 \
  --level action \
  --sparse-threshold 0.001

csh prompt \
  --prior-json outputs/fr_st/causal_prior.json \
  --compressed-pkl path/to/smartgen_compressed.pkl \
  --device-info-json path/to/device_info.json \
  --original-context winter \
  --new-context spring \
  --out-prompt outputs/fr_st/causal_prompt.txt

csh filter \
  --prior-json outputs/fr_st/causal_prior.json \
  --generated-pkl path/to/smartgen_generated.pkl \
  --out-pkl outputs/fr_st/generated_causal_kept.pkl \
  --out-scores outputs/fr_st/causal_scores.json \
  --min-coverage 0.5
```

最后将 `generated_causal_kept.pkl` 与训练集按实验方案合并，调用原 SmartGuard 的训练脚本完成异常检测模型训练与测试。
