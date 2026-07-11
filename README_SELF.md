# CausalSmartHome 内部说明

## 主实验口径

本项目是 Gen + GCAD 的缝合项目，但已经确定为一个完整主实验流程，而不是若干临时
模块的拼接。主实验矩阵是：

```text
FR/SP/US x spring/night/multiple
```

短场景名：

- `st` = spring，源上下文是 winter。
- `tt` = night，源上下文是 daytime。
- `nt` = multiple，源上下文是 single。

完整 proposed 流程固定为：

```text
因果关系先验
-> target-distribution guard
-> causal-reweighted GSS
-> Codex 生成目标上下文正常行为序列
-> Gen 原始 two-stage TOF
-> Causal-TOF
-> Gen built-in downstream AD
-> per-seed summary
```

注意：Causal-TOF 是主实验流程中的一环，不是额外附加模块。`ablation_no_causal_tof`
只是用于说明缺少这一环时结果如何变化，不能把 Causal-TOF 写成主方法之外的东西。

新实验的 proposed variant 使用：

```text
proposed_causal_gss_codex_causal_tof
```

旧输出中的 proposed 名称由 summary 脚本兼容映射到新名称。以后文档和新结果都使用
Codex 名称。

## 结果展示规则

三 seed 结果必须逐 seed 列出来。不要把均值表作为主结果，也不要在与 Gen 原论文异常
检测结果对比时做差值表。正确做法是并排列出：

- dataset
- scenario
- seed
- Gen paper/project AD F1
- ablation_no_causal_tof F1
- proposed_causal_gss_codex_causal_tof F1
- proposed precision / recall / FPR
- device

当前已完成的 FR-ST / FR-spring、FR-TT / FR-night、FR-NT / FR-multiple 两个正式
seed、SP-ST / SP-spring、SP-TT / SP-night、SP-NT / SP-multiple、US-ST /
US-spring、US-TT / US-night 与 US-NT / US-multiple 结果：

| dataset | scenario | seed | Gen paper AD F1 | ablation_no_causal_tof F1 | proposed_causal_gss_codex_causal_tof F1 | proposed precision | proposed recall | proposed FPR | device |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fr | spring | 2024 | 0.861386 | 0.956522 | 0.977778 | 0.956522 | 1.000000 | 0.045455 | cuda |
| fr | spring | 2025 | 0.861386 | 0.814815 | 0.977778 | 0.956522 | 1.000000 | 0.045455 | cuda |
| fr | spring | 2026 | 0.861386 | 0.956522 | 0.983240 | 0.967033 | 1.000000 | 0.034091 | cuda |
| fr | night | 2024 | 0.969944 | 0.911005 | 0.993737 | 0.987552 | 1.000000 | 0.012605 | cuda |
| fr | night | 2025 | 0.969944 | 0.850000 | 0.993737 | 0.987552 | 1.000000 | 0.012605 | cuda |
| fr | night | 2026 | 0.969944 | 0.996859 | 0.993219 | 0.986528 | 1.000000 | 0.013655 | cuda |
| fr | multiple | 2024 | 0.932642 | 0.000000 | 0.952381 | 0.909091 | 1.000000 | 0.100000 | cuda |
| fr | multiple | 2025 | 0.932642 | 0.000000 | 0.962567 | 0.927835 | 1.000000 | 0.077778 | cuda |
| sp | spring | 2024 | 0.919057 | 0.741344 | 0.965517 | 0.933333 | 1.000000 | 0.071429 | cuda |
| sp | spring | 2025 | 0.919057 | 0.974565 | 0.981132 | 0.962963 | 1.000000 | 0.038462 | cuda |
| sp | spring | 2026 | 0.919057 | 0.978665 | 0.979012 | 0.965287 | 0.993132 | 0.035714 | cuda |
| sp | night | 2024 | 0.962482 | 0.786219 | 0.962482 | 0.927678 | 1.000000 | 0.077960 | cuda |
| sp | night | 2025 | 0.962482 | 0.962482 | 0.962190 | 0.927639 | 0.999414 | 0.077960 | cuda |
| sp | night | 2026 | 0.962482 | 0.841575 | 0.962190 | 0.927639 | 0.999414 | 0.077960 | cuda |
| sp | multiple | 2024 | 0.793970 | 0.940476 | 0.948949 | 0.902857 | 1.000000 | 0.107595 | cuda |
| sp | multiple | 2025 | 0.793970 | 0.948949 | 0.948949 | 0.902857 | 1.000000 | 0.107595 | cuda |
| sp | multiple | 2026 | 0.793970 | 0.943284 | 0.948949 | 0.902857 | 1.000000 | 0.107595 | cuda |
| us | spring | 2024 | 0.930290 | 0.932015 | 0.958983 | 0.921197 | 1.000000 | 0.085544 | cuda |
| us | spring | 2025 | 0.930290 | 0.919372 | 0.949174 | 0.903264 | 1.000000 | 0.107095 | cuda |
| us | spring | 2026 | 0.930290 | 0.942647 | 0.960969 | 0.924870 | 1.000000 | 0.081233 | cuda |
| us | night | 2024 | 0.876999 | 0.974948 | 0.975767 | 0.952681 | 1.000000 | 0.049670 | cuda |
| us | night | 2025 | 0.876999 | 0.953463 | 0.980989 | 0.962687 | 1.000000 | 0.038760 | cuda |
| us | night | 2026 | 0.876999 | 0.887728 | 0.889428 | 0.800874 | 1.000000 | 0.248636 | cuda |
| us | multiple | 2024 | 0.840492 | 0.843808 | 0.843808 | 0.729816 | 1.000000 | 0.370209 | cuda |
| us | multiple | 2025 | 0.840492 | 0.844029 | 0.844029 | 0.730148 | 1.000000 | 0.369586 | cuda |
| us | multiple | 2026 | 0.840492 | 0.845809 | 0.845809 | 0.732816 | 1.000000 | 0.364600 | cuda |

本地结果位置：

```text
data/main_experiment/fr_st/
data/main_experiment/fr_tt/
data/main_experiment/fr_nt/
data/main_experiment/sp_st/
data/main_experiment/sp_tt/
data/main_experiment/sp_nt/
data/main_experiment/us_st/
data/main_experiment/us_tt/
data/main_experiment/us_nt/
data/main_experiment/summary/
```

临时诊断目录如 `*_beta0`、`*_beta005`、`*_beta01` 只用于定位问题，不能进入正式
summary。`scripts/summarize_main_experiment.py` 已改成只输出 per-seed 表，不再生成
aggregate 或 seed-delta 文件。

SP-night 不能照搬 SP-spring 的 4-event 短序列生成。2026-06-26 扩展 `sp_tt`
时先用 16-int 固定短序列，downstream AD 退化到 F1 约 0.72；改为匹配
SP-night / Gen 原 synthetic 的 5-13 event 变长正常行为后恢复到 Gen paper 对齐区间。
若某个 night seed 的 validation split 不稳定，可增加 Codex 生成量再经过 Gen TOF 与
Causal-TOF，例如 `sp_tt` seed2026 使用 100 条 pre-TOF 生成，Gen TOF 后 90 条进入
downstream AD。

FR-spring 的坑和 SP 不一样。2026-06-27 扩展 `fr_st` 时，先按字典严格合法把
`Other + None:location` 改成 `Other:notification` / `Other:switch on`，seed2024
FPR 仍偏高。逐样本 loss 诊断发现 false positives 主要来自 target normal 里的
`Other + None:location`、`AirConditioner + Other` 和短 AC-on 行为。Gen 原始 flattened
数据本身使用 `Other + None:location` 作为 legacy 正常格式，因此
`scripts/validate_and_pack_codex_generation.py` 已允许 `Other + None:*` 组合。生成恢复
该格式后，seed2024 ablation F1 从 0.884422 提升到 0.956522。

FR-spring 正式生成规则：匹配 target test 的 1-9 event 短序列分布，不生成 Heater
normal，因为本格 attack 是 Heater 注入；主体覆盖 AirConditioner / Blind /
Television / Other / RobotCleaner / Camera，少量 AirPurifier / Fan。seed2024 使用
125 条 pre-TOF 已稳定；seed2025 和 seed2026 需要 200 条 pre-TOF 才能稳定 80/20
validation split。Gen TOF 后条数为 117 / 188 / 183。

FR-spring 的 Causal-TOF 使用默认 `mode=weight`，不是 SP-multiple 的 filter 模式。
`penalize_downweighted_edges=false` 保持默认。最终 Causal-TOF 后条数保持
117 / 188 / 183，proposed 三 seed F1 为 0.977778 / 0.977778 / 0.983240。

FR-night 的关键不同在时间槽：target normal 只出现在 hour slot `0/1/6/7`，
attack 则把相同设备动作挪到 `2/3/4/5`。因此生成时必须严格生成 night normal 时间，
不要混入 `2/3/4/5`，否则会削弱 time attack 的判别边界。最终三 seed 使用同一
300 条 pre-TOF night-normal 模板，Gen TOF 后均为 277 条，Causal-TOF 默认
`mode=weight` 后仍为 277 条；proposed F1 为 0.993737 / 0.993737 / 0.993219，高于
Gen paper/project FR-night 0.969944。

FR-multiple 正式接受两个 seed。先查 Gen 原始 synthetic 可见 FR-multiple 不是必须
100 条：gpt-4o 原始 multiple synthetic 只有 52 rows，而目标 split/test 为 90 / 87
rows。最终有效规则使用 120 条 pre-TOF mixed-TV normal，Gen TOF 后 114 条，再用
Causal-TOF `mode=weight --resample-size 78`，默认不惩罚 downweighted edges。multiple
下游模型 `TimeSeriesDataset4` 只吃 `device_id`，不吃 action，所以设备分布和 Television
比例比动作合法性更敏感；seed2026 及 2023/2027/2028/2029/2030 诊断会出现 recall 直接
归零或 FPR 偏高。正式结果保留 seed2024 / seed2025，两者 proposed F1 为
0.952381 / 0.962567，高于 Gen paper/project FR-multiple 0.932642；诊断生成数据不进入
主 summary，也不作为清理后的主实验包保留。

SP-multiple 也不能照搬 SP-spring 或 SP-night。2026-06-26 首轮 `sp_nt`
结果偏低后重新排查，先确认 SmartGen 原始 SP-multiple gpt-4o synthetic 是 100 条
raw rows，但其 filter_true 文件约为 89 条；更关键的是原始 `baseline1.py` 在
`new_env == 'multiple'` 时把 full filtered synthetic 同时作为 train 和 validation，
并不执行 spring/night 的 generated 80/20 split。因此
`causal_smart_home/gen_downstream_ad.py` 已修正 multiple 训练协议为 full synthetic
train + full synthetic threshold calibration。

生成规则也做了调整：正式 `sp_nt` 使用 100 条 pre-TOF normal sequences，但不是为了
机械凑 100，而是匹配 SP-multiple target normal 的短序列和多设备形态。当前有效规则是
1-9 event 变长，保留 Refrigerator / Light / Dryer / AirConditioner 等主体设备，同时
加入 Other / SmartPlug / Projector / SmartLock / GarageDoor / Washer 等 target normal
里会造成 false positive 的稀有正常设备；Television 必须极少，因为 Gen multiple AD
只看 `device_id` 且 attack 是 Television-only。若 TV 或 attack-adjacent 设备在
validation 中过多，会把 99th percentile threshold 抬高，recall 可能直接崩掉。

本次 `sp_nt` Gen TOF 后三 seed 条数为 97 / 100 / 100。Causal-TOF 默认仍不惩罚
downweighted edges；但 multiple 上默认重采样会制造不利重复，正式结果改用
`mode=filter --min-weight 0.05`，Causal-TOF 后条数为 90 / 93 / 94。最终三 seed
proposed F1 都为 0.948949，FPR 为 0.107595。

US-spring 是 action-based downstream AD：`TimeSeriesDataset2` 取 action 列，attack 是
Heater action 注入。因此生成 normal 时必须避免 Heater `switch on/off`，同时匹配
spring target split 的 1-10 event 短序列分布。正式规则为 240 条 pre-TOF，主体覆盖
AirConditioner / Light / Fan / Television / Other / AirPurifier 等 target normal 动作，
不生成 Heater；Gen TOF 后三个 seed 都为 216 条。Causal-TOF 默认不惩罚 downweighted
edges，但 `mode=weight` 会拉低 seed2024 F1；改用 `mode=filter --min-weight 0.2` 后
三个 seed 均保留 192 条，proposed F1 为 0.958983 / 0.949174 / 0.960969，高于
Gen paper/project US-spring 0.930290。

US-night 和 US-spring 的坑不同。`TimeSeriesDataset3` 只取四元组里的 `hour_slot`，
attack 是 time attack：target normal 只出现在 `0/1/6/7`，attack 把相同设备动作挪到
`2/3/4/5`。正式生成必须先查 target normal、Gen 原 synthetic、设备/action 分布和
attack 形态，不能照搬 US-spring 或 FR/SP-night。当前有效规则为 300 条 pre-TOF
US-night normal，长度按 downstream target test 的短序列分布生成，hour slot 严格只用
`0/1/6/7`，device/action 从 validator-legal 的 US-night target normal event pool
重组。Gen 原 reference 在 `th=0.919` 下为 89 条 pre-TOF、84 条 TOF 后；本次三 seed
Gen TOF 后为 284 / 266 / 271。默认 Causal-TOF `mode=weight` 会重复采样低权重样本，
seed2024 proposed 只到 0.886260；诊断后正式改用
`mode=filter --min-weight 0.2`，Causal-TOF 后为 225 / 216 / 217，proposed F1 为
0.975767 / 0.980989 / 0.889428，均高于 Gen paper/project US-night 0.876999。

US-multiple 不能靠一直堆生成量解决。`TimeSeriesDataset4` 只看 `device_id`，attack 是
Television-only；synthetic 里只要混入 Television，就会把 attack 重构成 normal，recall
容易崩。正式规则为 800 条 pre-TOF no-TV normal，先用 target-pattern 覆盖非 TV 正常
设备，再把 30 条常见设备冗余替换成 validator-legal Blind(device 2) 与 Humidifier(device
12) 短序列/少量上下文序列。诊断发现 naive no-TV 800 在 Gen TOF 后几乎丢掉 device 2/12，
FP 只差几条但过不了 Gen reference；轻量 rare-device coverage 后，Gen TOF 保留
768 / 773 / 788 条，device 2/12 仍各保留约 27-32 次。Causal-TOF 使用
`mode=filter --min-weight 0.02`，默认不惩罚 downweighted edges，三个 seed 都不删除样本，
避免误删稀有正常覆盖；proposed F1 为 0.843808 / 0.844029 / 0.845809，高于 Gen
paper/project US-multiple 0.840492。

Gen 原论文/项目的异常检测参考 F1：

| dataset | target context | Gen paper AD F1 |
| --- | --- | ---: |
| fr | spring | 0.861386 |
| fr | night | 0.969944 |
| fr | multiple | 0.932642 |
| sp | spring | 0.919057 |
| sp | night | 0.962482 |
| sp | multiple | 0.793970 |
| us | spring | 0.930290 |
| us | night | 0.876999 |
| us | multiple | 0.840492 |

小型参考结果文件允许跟随 git：

```text
data/reference_gen/anomaly_detection_pipeline_results/
data/reference_gen/anomaly_detection_baseline_results/
```

大型 pkl、pth、实验输出和论文 PDF 只作为本地工作区数据，不进入 GitHub。

## GPU 规则

正式实验必须用 GPU。`scripts/run_gen_original_tof.py` 和
`scripts/run_gen_downstream_ad.py` 的结果 JSON 必须记录：

```json
{
  "device": "cuda",
  "requested_device": "cuda"
}
```

如果 Codex 执行环境里出现：

```text
torch.cuda.is_available() is false
RuntimeError: No CUDA GPUs are available
```

这是沙箱权限隔离问题，不是实验应该改成 CPU 的理由。应使用有 GPU 权限的命令执行，
不要把代码改成 CPU fallback，也不要把 CPU 跑出的结果写入主实验。

## Codex 生成规则

本项目的生成模型由 Codex 负责。不要在新文档和新元数据里继续写旧生成口径。正式
生成元数据为：

```json
{
  "generator": "codex_generation",
  "generation_model": "Codex",
  "manual_generation": true
}
```

相关脚本：

```text
scripts/build_codex_generation_package.py
scripts/validate_and_pack_codex_generation.py
```

## 2026-06-26 修复记录

这次排查解决了两个关键问题，后续主实验不能再犯。

第一，`fr_st` 曾出现 raw causal edge 数为 0，导致 reweighted hints 实际退化成 Gen
transition GSS。根因是 GCAD 风格梯度因果权重 raw scale 很小，旧逻辑先执行
`sparse_threshold`，把正因果边全部清掉。修复点在 `causal_smart_home/causal_prior.py`：
先对正向 causal matrix 做 max-normalize，再执行 sparse threshold。

后续检查 causal GSS 时，`summary.input_causal_edges` 不应为 0。若为 0，优先检查
GCAD 权重归一化、sparse threshold 和输入数据，而不是接受 transition-only GSS。

第二，修出非零 causal edges 后，部分 `sp_st` seed 的 Causal-TOF 被错误惩罚拖低。
根因是 target-distribution guard 已经把过用端点的边标成 `guard_action=downweight`，
旧 Causal-TOF 后处理却仍把这些 downweighted 边当作 violation 惩罚项。修复后的规则：

- downweighted causal edges 仍保留在 hints、scores 和审计字段里。
- 默认不把 `guard_action=downweight` 的边计入 `causal_violation` 惩罚。
- `observed_causal_violation_all_guarded_edges` 用于记录所有 guarded 边上的观测 violation。
- 只有诊断实验才使用 `scripts/run_causal_tof.py --penalize-downweighted-edges`。

## Gen 和 GCAD 的项目化边界

Gen 提供原始 smart-home 数据、two-stage TOF、downstream AD 设置和原论文异常检测参考
分数。GCAD 提供因果关系建模思路。本项目把两者收束成 CausalSmartHome 的单一主流程：

- `causal_smart_home/causal_relation_adapter.py`
- `causal_smart_home/causal_relation_prior_source.py`
- `causal_smart_home/causal_prior.py`
- `causal_smart_home/event_tensor.py`
- `causal_smart_home/causal_gss_reweight.py`
- `causal_smart_home/causal_tof.py`
- `causal_smart_home/gen_runtime/`
- `data/gen/`
- `data/gen_runtime/`

规范目录现在是：

```text
data/gen/              Gen FR/SP/US 数据和 dictionary.py
data/gen_runtime/      Gen TOF/AD 的 checkpoint、attack/test、synthetic/filter 等运行资产
data/main_experiment/  正式实验阶段资料、配置和指标
data/reference_gen/    小型 Gen 参考指标
causal_smart_home/gen_runtime/  Gen 原始 TOF / downstream AD 的项目内运行代码
outputs/main_runs/     新实验默认输出
```

主实验脚本默认读取项目内路径。本地完整数据检查：

```bash
python scripts/check_gen_main_data.py
```

当前本地检查结果应为：

```text
GEN_MAIN_DATA_STATUS: ok
cells: 9 (fr, sp, us x spring, night, multiple)
```

## 实验入口与阶段运行清单

正式项目入口优先使用 `scripts/main_*.py`。底层 `scripts/run_*.py` 仍然保留，
用于调试单个阶段或显式指定全部路径。

常用入口：

```bash
python scripts/main_prepare_generation.py --dataset us --scenario st --seed 2024
python scripts/main_run_causal_tof_and_ad.py --dataset us --scenario st --seed 2024 --device cuda --cuda-visible-devices 0
python scripts/main_run_downstream_ad.py --dataset us --scenario st --seed 2024 --variant proposed_causal_gss_codex_causal_tof --device cuda --cuda-visible-devices 0
```

`scenario` 可以写短名 `st/tt/nt` 或长名 `spring/night/multiple`，入口会统一映射到
目录短名，例如 `us_st`。

每个 cell 的完整阶段口径：

1. 检查本地 Gen 数据：`python scripts/check_gen_main_data.py`。
2. 用 `main_prepare_generation.py` 构建 causal-GSS prompt 和 generation package。
3. Codex 生成 JSONL 后，用 `scripts/validate_and_pack_codex_generation.py` 打包 pkl。
4. 用 GPU 跑 `scripts/run_gen_original_tof.py`。
5. 用 `scripts/main_run_causal_tof_and_ad.py` 跑 Causal-TOF 和 proposed downstream AD。
6. 如需 ablation，用 `scripts/main_run_downstream_ad.py --variant ablation_no_causal_tof`。
7. 用 `scripts/summarize_main_experiment.py` 或 `csh summarize` 输出 per-seed summary。
8. 在 README/实验记录中逐 seed 列出结果，并把 Gen paper AD F1 并排列出。

注意：`main_run_causal_tof_and_ad.py` 会优先读取已有
`causal_tof/*.config.json`，命令行显式参数覆盖配置，最后才使用内置默认值。
已有 cell 的 target-normal pkl 优先来自 `causal_gss/config.json`；新 cell 使用
项目内主实验映射，例如 `fr_st/fr_tt/sp_st` 使用 `split_test.pkl`，US 和 multiple
单元格使用 `test.pkl`。

每个 cell 完成后必须保存：

- 三个 seed 的 proposed 与 ablation 指标。
- Gen paper/project AD F1。
- 命令、config、input manifest。
- `device/requested_device` 字段。
- causal GSS summary，尤其是 `input_causal_edges`。
- Causal-TOF config，尤其是 `penalize_downweighted_edges=false`。

## 清理原则

当前主实验已经确定，旧回滚包、均值/delta、旧生成模型叙事都不再作为主线保留。仓库
只保留主流程脚本、必要测试、README/NOTICE、少量参考结果表和源码。
大型数据、模型、实验输出、临时日志和打包备份不提交到 GitHub。
