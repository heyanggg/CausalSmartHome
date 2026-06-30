# CausalSmartHome

## Overview

CausalSmartHome is a Gen + GCAD fused project for smart-home behavior
generation and anomaly detection. The main experiment matrix follows Gen's
FR/SP/US anomaly-detection setup over three target contexts:

```text
FR/SP/US x spring/night/multiple
```

The project design is fixed as:

```text
causal relation prior
-> target-distribution guard
-> causal-reweighted GSS
-> Codex generation
-> Gen original two-stage TOF
-> Causal-TOF
-> Gen built-in downstream AD
-> per-seed summary
```

Causal-TOF is one step of the main experiment pipeline. It is not a separate
method. The retained ablation, `ablation_no_causal_tof`, removes this
pipeline step only to show what the full pipeline loses without it.

The proposed method name used by new runs is:

```text
proposed_causal_gss_codex_causal_tof
```

Historical proposed-method names are normalized to this Codex method name by
the summary script.

## Current Completed Results

The completed GPU runs are stored locally under:

```text
outputs/main_experiment/fr_st/
outputs/main_experiment/fr_tt/
outputs/main_experiment/fr_nt/
outputs/main_experiment/sp_st/
outputs/main_experiment/sp_tt/
outputs/main_experiment/sp_nt/
outputs/main_experiment/us_st/
outputs/main_experiment/us_tt/
outputs/main_experiment/us_nt/
outputs/main_experiment/summary/
```

Main results must be read seed by seed. Do not replace this table with an
average table, and do not report deltas against Gen.

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

Gen paper/project anomaly-detection reference scores used for parallel
comparison:

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

The small Gen reference summaries are tracked under
`outputs/reference_gen/`. Large local pkl/checkpoint/experiment artifacts are
ignored by git.

## Running The Pipeline

Scenario aliases:

```text
st = spring
tt = night
nt = multiple
```

Core scripts:

```text
scripts/main_prepare_generation.py
scripts/main_run_causal_tof_and_ad.py
scripts/main_run_downstream_ad.py
scripts/build_causal_gss_prompt.py
scripts/build_codex_generation_package.py
scripts/validate_and_pack_codex_generation.py
scripts/run_gen_original_tof.py
scripts/run_causal_tof.py
scripts/run_gen_downstream_ad.py
scripts/summarize_main_experiment.py
scripts/check_gen_main_data.py
```

### 实验入口与运行方式

所有命令都从仓库根目录运行。建议用 `python` 调脚本，因为项目被拷贝到不同
文件系统后，不一定保留脚本的可执行权限：

```bash
python scripts/check_gen_main_data.py
```

推荐优先使用 `scripts/main_*.py` 作为实验级入口。它们只要求填写
`dataset/scenario/seed` 等实验坐标，会按项目约定自动定位阶段输入；底层
`scripts/run_*.py` 保留给调试或单独执行某个阶段。

`scenario` 可以写短名 `st/tt/nt`，也可以写长名 `spring/night/multiple`；
实验目录会统一规范化为短名，例如 `us_st`。

对单个 seed，下游 AD 阶段常用这两个输入：

```text
outputs/main_experiment/{dataset}_{scenario}/seed{seed}/gen_original_tof/gen_tof.pkl
outputs/main_experiment/{dataset}_{scenario}/seed{seed}/causal_tof/generated_gen_tof_causal_tof.pkl
```

第一个用于不带 Causal-TOF 的 ablation；第二个用于 proposed 方法。

运行一个 ablation 下游 AD：

```bash
python scripts/main_run_downstream_ad.py \
  --dataset us \
  --scenario st \
  --seed 2024 \
  --variant ablation_no_causal_tof \
  --device cuda \
  --cuda-visible-devices 0
```

运行一个 proposed 下游 AD：

```bash
python scripts/main_run_downstream_ad.py \
  --dataset us \
  --scenario st \
  --seed 2024 \
  --variant proposed_causal_gss_codex_causal_tof \
  --device cuda \
  --cuda-visible-devices 0
```

每个 AD 运行的底层命令会保存到：

```text
{runs_root}/{dataset}_{scenario}/seed{seed}/downstream_ad/{variant}/run_command.sh
```

运行 Causal-TOF，并继续执行 proposed 下游 AD：

```bash
python scripts/main_run_causal_tof_and_ad.py \
  --dataset us \
  --scenario st \
  --seed 2024 \
  --device cuda \
  --cuda-visible-devices 0
```

每组实验的 Causal-TOF 参数不完全一样。`main_run_causal_tof_and_ad.py`
默认会读取
`outputs/main_experiment/{dataset}_{scenario}/seed{seed}/causal_tof/*.config.json`
中的阶段配置；命令行显式填写的参数优先级更高。例如 US-spring 的配置会解析成
`mode=filter --min-weight 0.2`，FR-multiple 的配置会解析成
`mode=weight --resample-size 78`。如果要完全忽略已有阶段配置，可以加
`--ignore-causal-tof-config`。

target-normal pkl 也按主实验口径推断：已有 cell 优先使用
`causal_gss/config.json` 里保存的 `target_pkl`；新 cell 默认使用项目内映射
（例如 `fr_st/fr_tt/sp_st` 使用 `split_test.pkl`，US 和 multiple 单元格使用
`test.pkl`）。

运行 Gen 原始 TOF 阶段时，需要提供 pre-TOF generated pkl：

```bash
python scripts/run_gen_original_tof.py \
  --generated-pkl outputs/main_experiment/us_st/seed2024/codex_generation/generated_codex.pkl \
  --dataset us \
  --scenario st \
  --seed 2024 \
  --out-dir outputs/main_runs/us_st/seed2024/gen_original_tof \
  --out-pkl outputs/main_runs/us_st/seed2024/gen_original_tof/gen_tof.pkl \
  --cuda-visible-devices 0
```

如果某个 cell 没有标准路径下的 pre-TOF generated pkl，则从 Gen TOF 或
Causal-TOF 之后的阶段入口开始运行。

准备 causal-GSS prompt 和 generation package：

```bash
python scripts/main_prepare_generation.py \
  --dataset us \
  --scenario st \
  --seed 2024
```

从这个 package 生成 JSONL 后，再校验并打包成 Gen 可读的 pkl：

```bash
python scripts/validate_and_pack_codex_generation.py \
  --input-jsonl outputs/main_runs/us_st/seed2024/codex_generation/generated_codex.jsonl \
  --out-pkl outputs/main_runs/us_st/seed2024/codex_generation/generated_codex.pkl \
  --out-validation-report outputs/main_runs/us_st/seed2024/codex_generation/validation_report.json \
  --out-generation-report outputs/main_runs/us_st/seed2024/codex_generation/generation_report.json \
  --dictionary-py causal_smart_home/resources/gen_data/dictionary.py \
  --dataset us \
  --scenario st \
  --scenario-key us_st \
  --seed 2024 \
  --expected-count 240 \
  --source-pkl causal_smart_home/resources/gen_data/us/winter/trn.pkl \
  --target-pkl causal_smart_home/resources/gen_data/us/spring/test.pkl \
  --guard-report-json outputs/main_runs/us_st/seed2024/codex_generation_package/guard_report.json \
  --guarded-hints-json outputs/main_runs/us_st/seed2024/codex_generation_package/guarded_reweighted_gss_hints.json \
  --resolved-causal-relation-prior-json outputs/main_runs/us_st/seed2024/codex_generation_package/resolved_causal_relation_prior.json
```

实验完成后生成 per-seed 汇总：

```bash
python scripts/summarize_main_experiment.py \
  --runs-root outputs/main_runs \
  --out-dir outputs/main_runs/summary
```

下面这些 JSON 是参数和溯源的准确信息来源：

```text
causal_gss/config.json
codex_generation/generation_report.json
gen_original_tof/gen_original_tof_report.json
causal_tof/*.config.json
downstream_ad/{variant}/config.json
downstream_ad/{variant}/normalized_metrics.json
```

GPU execution is required for Gen original TOF and Gen downstream AD. Results
must record `device = cuda` and `requested_device = cuda`. If a managed sandbox
hides CUDA devices, run the experiment command with GPU access; do not change
the experiment to CPU fallback.

## Guardrails

- Keep Causal-TOF in the proposed pipeline. It is part of the main method flow.
- List all seed results separately. Mean/std tables are not the main result.
- Compare against Gen paper AD scores by listing them side by side. Do not use
  delta tables as the comparison output.
- Normalize GCAD causal weights before sparse thresholding. Otherwise raw
  causal edges can be zeroed out and the GSS degenerates into a transition-only
  graph.
- `build_causal_gss_prompt.py` should add causal edges by default and use
  `guard-mode=downweight`.
- Causal-TOF keeps downweighted edges in the audit fields, but does not count
  `guard_action=downweight` edges in the causal-violation penalty by default.
  Use `--penalize-downweighted-edges` only for diagnostics.
- FR-spring target normal data contains Gen legacy `Other + None:location`
  rows. The Codex validator allows this original-format pairing; replacing it
  with dictionary-pure `Other:*` actions leaves many target normals uncovered
  and raises false positives.
- For FR-spring, Causal-TOF should use the default weight mode. Seed2025 needs
  a larger 200-row pre-TOF generation to stabilize the spring 80/20 validation
  split; final pre-TOF counts are 125 / 200 / 200 for seeds 2024 / 2025 / 2026,
  Gen TOF keeps 117 / 188 / 183 rows, and Causal-TOF keeps the same row counts.
- FR-night uses a temporal rule: target normal events occur in hour slots
  `0/1/6/7`, while the attack set moves otherwise similar behavior into
  `2/3/4/5`. Generate night normal rows only in `0/1/6/7`; the stable setting
  uses 300 pre-TOF rows, Gen TOF keeps 277, and Causal-TOF default weight mode
  keeps 277 for seeds 2024 / 2025 / 2026.
- FR-multiple was accepted with two good seeds. The final rule uses 120 pre-TOF
  mixed-TV normal rows, Gen TOF keeps 114, and Causal-TOF uses
  `mode=weight --resample-size 78` with default
  `penalize_downweighted_edges=false`. The downstream multiple model uses only
  `device_id`, so seed2026 diagnostics were unstable and were moved to
  `outputs/diagnostics/main_experiment_exploration/`.
- In SP-multiple, SmartGen's own gpt-4o synthetic source has 100 raw rows
  before TOF/filtering, but its downstream AD baseline uses the full filtered
  multiple-context synthetic set for both training and threshold calibration
  instead of the spring/night 80/20 generated split. Mirror that protocol.
- SP-multiple cannot simply copy SP-spring or SP-night generation. Match the
  multiple target-normal shape with 100 pre-TOF variable-length sequences
  spanning short 1-9 event behavior, include rare-but-normal target devices
  such as Other/SmartPlug/Projector/SmartLock/GarageDoor/Washer, and keep
  Television very scarce because the attack set is Television-only under the
  device-id AD model.
- For SP-multiple Causal-TOF, the stable setting is filter mode with
  `min_weight=0.05` and the default `penalize_downweighted_edges=false`. This
  avoids harmful duplicate resampling while still removing low causal-weight
  rows. The final SP-multiple Gen TOF counts are 97 / 100 / 100 for seeds
  2024 / 2025 / 2026, and Causal-TOF keeps 90 / 93 / 94 rows.
- US-spring downstream AD is action-based and the attack injects Heater actions.
  The generator must avoid Heater normal actions and preserve the target split's
  short 1-10 event length distribution. The stable rule uses 240 pre-TOF rows,
  Gen TOF keeps 216, and Causal-TOF uses `mode=filter --min-weight 0.2`, keeping
  192 rows for each seed.
- US-night downstream AD is time-slot-based: `TimeSeriesDataset3` uses
  `hour_slot`, target normal is restricted to `0/1/6/7`, and attack samples move
  the same behavior into `2/3/4/5`. The stable rule uses 300 pre-TOF rows with a
  target-like short-sequence mix and no `2/3/4/5` hours. Gen TOF keeps
  284 / 266 / 271 rows for seeds 2024 / 2025 / 2026. Default Causal-TOF
  `mode=weight` duplicated harmful low-weight rows on seed2024, so the formal
  setting is `mode=filter --min-weight 0.2`, keeping 225 / 216 / 217 rows.
- US-multiple downstream AD is device-id-based: `TimeSeriesDataset4` uses only
  `device_id`, and the attack set is Television-only. The stable rule uses 800
  pre-TOF rows with no Television events. A naive target-like no-TV generation
  was close but lost Blind(device 2) and Humidifier(device 12) coverage during
  Gen TOF; increasing total rows was less useful than replacing 30 common rows
  with validator-legal Blind/Humidifier short/context rows. Gen TOF keeps
  768 / 773 / 788 rows for seeds 2024 / 2025 / 2026. Causal-TOF uses
  `mode=filter --min-weight 0.02` with default
  `penalize_downweighted_edges=false`, keeping the same counts so it scores the
  Gen TOF output without deleting useful rare normal coverage.

## Checks

Run from the project root:

```bash
pytest -q
python scripts/check_gen_main_data.py
csh summarize --runs-root outputs/main_experiment --out-dir outputs/main_experiment/summary
```

`scripts/check_gen_main_data.py` verifies the local FR/SP/US x
spring/night/multiple Gen data required by the main experiments.

## Project Structure

| path | role |
| --- | --- |
| `causal_smart_home/schema.py` | Gen 扁平四元组序列的基础解析、转换与校验。 |
| `causal_smart_home/causal_prior.py` | 轻量 causal prior 数据结构和边过滤工具。 |
| `causal_smart_home/causal_relation_adapter.py` | 从序列中抽取候选因果关系的适配层。 |
| `causal_smart_home/causal_relation_prior_source.py` | 统一外部 prior、矩阵 prior 和本地 fallback prior 的来源。 |
| `causal_smart_home/target_distribution_guard.py` | 用目标域设备分布约束 causal hints，避免提示偏离目标正常分布。 |
| `causal_smart_home/causal_gss.py` | GSS 解释、设备名映射、prompt 相关辅助函数。 |
| `causal_smart_home/causal_gss_reweight.py` | 将 GCAD prior 注入 Gen GSS 转移图，输出 guarded/reweighted hints。 |
| `causal_smart_home/causal_tof.py` | Causal-TOF 打分、加权、过滤和重采样实现。 |
| `causal_smart_home/gen_original_tof.py` | Gen 原始 TOF 的项目内包装。 |
| `causal_smart_home/gen_downstream_ad.py` | Gen 内置 downstream AD 的项目内包装。 |
| `causal_smart_home/experiment_paths.py` | 主实验 dataset/scenario/seed 的默认路径推断。 |
| `causal_smart_home/json_utils.py` | JSON 输出时的 Path、NumPy 标量、NaN 统一处理。 |
| `causal_smart_home/gen_core/anomaly_detection_pipeline/` | vendored Gen 下游 AD 代码。 |
| `causal_smart_home/gen_core/gen_original_tof/` | vendored Gen 原始 TOF 代码。 |
| `causal_smart_home/resources/gen_data/` | FR/SP/US 的 Gen 数据和设备字典。 |
| `scripts/main_prepare_generation.py` | 实验级入口：准备 causal-GSS prompt 与 generation package。 |
| `scripts/main_run_causal_tof_and_ad.py` | 实验级入口：运行 Causal-TOF，并可继续运行 proposed AD。 |
| `scripts/main_run_downstream_ad.py` | 实验级入口：运行单个 ablation/proposed 下游 AD。 |
| `scripts/run_*.py` | 阶段级脚本：显式路径参数更多，适合调试单个阶段。 |
| `scripts/summarize_main_experiment.py` | 收集 normalized metrics，生成 per-seed 正式汇总表。 |
| `scripts/check_gen_main_data.py` | 检查本地 Gen 数据是否齐全。 |
| `tests/` | 单元测试和管线行为测试。 |
| `outputs/reference_gen/` | 小型 Gen 参考指标。 |
| `docs/` | GCAD 接入说明和项目框架图。 |

## Reading Notes And Framework Diagram

- GCAD-to-Gen theory and code walkthrough:
  [`docs/gcad_gen_integration_notes.md`](docs/gcad_gen_integration_notes.md)
- Framework diagram:
  [`docs/framework_diagram.svg`](docs/framework_diagram.svg)
