# Task 13: 更深层 GCAD-to-SmartGen Stage 4 接入记录

## 0. 边界和结论

本次实现的是 **GCAD-to-SmartGen glue**，不是新的 GCAD。代码没有删除旧 Stage 3 脚本，也没有覆盖 `outputs/gcad_gss/`。Stage 4 的新增输出统一写入：

```text
outputs/gcad_gss_stage4/
```

核心边界：

- 不重写 GCAD predictor、通道分离梯度、Granger-style causal matrix、`A - A^T` 稀疏化。
- 不用 transition count、相关系数、mutual information、随机梯度等冒充 GCAD causal matrix。
- 不把 compact fallback 声称为 official GCAD reproduction。
- 不默认 hard delete 生成样本；Causal-TOF 默认做 soft weighting。
- 不声称 GCAD edges 是真实物理因果。
- 不声称 raw GCAD prior 无条件提升 SmartGen。
- 不声称所有数据集都提升。
- 不把 source-context causal prior 强行无保护迁移到 target context。
- 核心结论是 **guarded causal prior**，不是 raw causal prior。

## 1. 当前 Stage 3 现状梳理

### 1.1 当前 GCAD prior 在哪里学习或读取

Stage 3 的 GCAD prior 主要通过：

```text
causal_smart_home/causal_gss.py
causal_smart_home/gcad_adapter.py
causal_smart_home/causal_prior.py
scripts/build_gcad_gss_prompt.py
```

`causal_gss.learn_device_gcad_prior()` 把 SmartGen/SmartGuard 的扁平四元组行为序列读成 `BehaviorSequence`，再用 `EventTensorizer(level="device")` 转为 event tensor，最后调用 `GCADAdapter.mine_event_prior()`。当前 `GCADAdapter` 对 event tensor 使用项目内已有 `GradientCausalMiner` compact fallback；它遵循 GCAD 的预测器训练、通道分离梯度、绝对梯度积分和对称稀疏化思想，但不是官方 GCAD 论文仓库的完整复现实验。

### 1.2 当前 causal hints 在哪里生成

Stage 3 的 prompt-only hints 由：

```text
scripts/build_gcad_gss_prompt.py
causal_smart_home/causal_gss.py
causal_smart_home/causal_prompt_adapter.py
```

生成。流程是学习 device-level prior、抽取 top causal edges、映射设备名、格式化成软约束文本，然后插入 SmartGen 原 prompt 的 GSS 附近。

### 1.3 当前 guarded-edge / downweighted-edge ablation 在哪里实现或汇总

旧 guarded/downweighted ablation 主要在 Stage 3 SP-ST 脚本中实现：

```text
scripts/run_stage3a_gcad_gss_sp_st.py
scripts/summarize_stage3a_sp_st_guarded_ablation.py
scripts/summarize_stage3b_sp_st_repeats.py
```

`run_stage3a_gcad_gss_sp_st.py` 内部有 `apply_target_overrepresented_edge_guard()`，根据 reference arm 与 target real pkl 的 device frequency 差异，对 overrepresented endpoint 做 edge removal 或 downweight。这是 prompt-only 层面的 guarded-edge ablation，还没有把 guarded causal edges 接入 GSS graph 的 edge score。

### 1.4 Stage 3A generation quality 输入输出

输入：source normal pkl、target normal/test pkl、原始 SmartGen prompt 或模板、GCAD-GSS prompt、Codex-calibrated/offline generation 设置、TOF 路径。

输出位于：

```text
outputs/gcad_gss/<tag>/
outputs/gcad_gss/*stage3a_summary.csv
outputs/gcad_gss/*stage3a_summary.json
outputs/gcad_gss/*stage3a_summary.md
```

质量指标包含 causal coverage、violation rate、low evidence rate、action/device/transition JS to target、TOF kept rate 等。

### 1.5 Stage 3B downstream AD 输入输出

输入：Stage 3A 生成/过滤后的 synthetic pkl、目标测试集/attack pkl、SmartGen anomaly detection pipeline wrapper 参数。

输出位于：

```text
outputs/gcad_gss/<scenario>_stage3b_ad/<tag>/metrics.json
outputs/gcad_gss/<scenario>_stage3b_ad/<tag>/metrics.csv
outputs/gcad_gss/<scenario>_stage3b_ad/<tag>/metrics.md
```

指标包含 precision、recall、F1、FPR、FNR、accuracy 等。

### 1.6 当前 prompt-only GCAD-GSS 的局限

Stage 3 只把 GCAD causal edges 作为 prompt 文本提示。它的局限是：

1. raw GCAD prior 与 target context distribution 可能冲突，尤其 source context 某设备很常见但 target context 中不常见时，会诱导过度生成。
2. GCAD edge 没有进入 SmartGen GSS transition graph 的 edge scoring，只是文字层面的 soft hint。
3. TOF 后没有使用 GCAD signal 做 soft ranking/weighting，只有较硬的 causal consistency filter 或 prompt-level ablation。
4. 对 SP-ST 类似 `* -> Television` 的 bias，prompt-only 方式容易把 source-context edge 直接带到 target context。

### 1.7 Stage 4 要做什么

Stage 4 将 prompt-only 升级为：

```text
GCAD causal matrix
  -> target-distribution guard
  -> guarded causal-reweighted GSS graph
  -> guarded causal-GSS prompt
  -> SmartGen generation
  -> causal-TOF soft weighting / verifier
  -> downstream AD evaluation
```

## 2. 新增模块设计

### 2.1 `gcad_prior_source.py`: GCAD prior source resolver

新增接口：

```python
def resolve_gcad_prior(
    prior_json=None,
    prior_matrix_path=None,
    source_pkl=None,
    out_dir=None,
    gcad_project_dir=None,
    adapter_mode="existing",
    level="device",
    lag=4,
    sparse_threshold=0.001,
    seed=2024,
) -> ResolvedGCADPrior:
    ...
```

优先级：

1. 有 `prior_json`：直接读取已有 `causal_prior.json` 或 Stage 4 `resolved_gcad_prior.json`。
2. 有 `prior_matrix_path`：读取 `.json/.npy/.csv/.txt` matrix 并标准化。
3. 没有 prior 但有 `source_pkl`：调用当前已有 `GCADAdapter.mine_event_prior()`，不在 resolver 里写新因果算法。

如果走当前 compact fallback，输出明确标注：

```json
"gcad_source": "existing_adapter_compact_fallback"
```

标准输出字段包含：

```json
{
  "gcad_source": "...",
  "level": "device",
  "lag": 4,
  "sparse_threshold": 0.001,
  "channels": ["d:1", "d:2"],
  "matrix": [[0, 0.9], [0, 0]],
  "top_causal_edges": [
    {"source": "d:1", "target": "d:2", "source_id": 0, "target_id": 1, "weight": 0.9, "lag": 4}
  ]
}
```

### 2.2 `target_distribution_guard.py`: Target-Distribution Guard

新增配置：

```python
@dataclass
class TargetDistributionGuardConfig:
    max_overuse_ratio: float = 1.25
    min_target_freq: float = 0.001
    eps: float = 1e-8
    mode: str = "suppress"  # suppress / downweight
    downweight_factor: float = 0.25
    endpoint_policy: str = "target"  # target / source_or_target / both
```

核心逻辑：如果某 causal edge 的 endpoint 在 prompt/source/generated arm 中相对 target normal 过度出现：

```text
observed_freq(device) / max(target_freq(device), min_target_freq) > max_overuse_ratio
```

则 suppress 或 downweight。输出的 guard report 会保存每条 edge 的 `raw_weight`、`guarded_weight`、`guard_action`、`guard_reason`、endpoint frequency 与 overuse ratio。

SP-ST Television bias 的解释：当 source/prompt arm 中 Television 的频率显著高于 target spring normal，`* -> Television` 或 `Television -> *` 类型 edge 会被标记。若 `endpoint_policy=target`，只有 target endpoint 是 Television 的边会被 suppress/downweight；若 `source_or_target`，Television 作为 source 或 target 都会触发。

### 2.3 `causal_gss_reweight.py`: Causal-Reweighted GSS Graph

新增：

```python
def build_device_transition_graph(sequences, device_name_map=None) -> dict:
    ...

def reweight_gss_edges(
    transition_edges,
    causal_edges,
    lambda_causal=1.0,
    mode="multiplicative",
    add_causal_edges=False,
    top_k=50,
) -> dict:
    ...
```

它不会重新计算 GCAD。`causal_edges` 必须来自 `ResolvedGCADPrior` 或 guard 后的 causal edges。

两种 reweighting：

```text
additive:
final_score = normalized_transition_score + lambda_causal * normalized_guarded_causal_strength

multiplicative:
final_score = normalized_transition_score * (1 + lambda_causal * normalized_guarded_causal_strength)
```

如果 `add_causal_edges=False`，只重加权已有 GSS transition edges；如果为 True，允许把强 GCAD edge 加入 graph，并标记：

```json
"edge_origin": "gcad_augmented"
```

### 2.4 `causal_tof.py`: Causal-TOF Soft Weighting

新增：

```python
def score_sequence_causal_tof(
    sequence,
    guarded_edges,
    target_distribution=None,
    reconstruction_loss=None,
    alpha_rec=1.0,
    beta_violation=1.0,
    gamma_dist=1.0,
    temperature=2.0,
) -> dict:
    ...
```

输出包含：

```json
{
  "causal_coverage": 0.8,
  "causal_violation": 0.2,
  "distribution_penalty": 0.1,
  "final_score": 0.3,
  "sample_weight": 0.55,
  "satisfied_edges": [],
  "violated_edges": [],
  "missing_edges": [],
  "decision": "weight"
}
```

默认不删除样本：

```text
sample_weight = exp(-temperature * final_score)
```

同时实现了 `rank / weight / filter` 三种模式，以及 downstream AD 不支持 sample weights 时的 weighted resampling fallback。fallback 会限制最大复制次数，避免数据爆炸。

## 3. 新增脚本

### 3.1 Prompt Builder

```bash
python scripts/build_guarded_causal_reweighted_gss_prompt.py \
  --source-pkl path/to/source_normal.pkl \
  --target-pkl path/to/target_normal.pkl \
  --prior-json path/to/causal_prior.json \
  --out-prompt outputs/gcad_gss_stage4/demo/prompt.txt \
  --out-prior-json outputs/gcad_gss_stage4/demo/resolved_gcad_prior.json \
  --out-guard-report outputs/gcad_gss_stage4/demo/guard_report.json \
  --out-reweighted-hints outputs/gcad_gss_stage4/demo/guarded_reweighted_gss_hints.json \
  --lambda-causal 1.0 \
  --reweight-mode multiplicative \
  --guard-mode suppress \
  --max-overuse-ratio 1.25 \
  --top-k 50 \
  --seed 2024
```

Prompt 中显式包含三层信息：

1. 原 SmartGen GSS transition hints。
2. raw GCAD causal hints。
3. guarded causal-reweighted GSS hints。

关键优先级文字：

```text
Use the guarded causal-reweighted GSS hints as the primary structural guidance.
Use raw GCAD causal hints only as weak background evidence.
If raw GCAD causal hints conflict with guarded reweighted hints, follow the guarded reweighted hints.
Do not over-generate devices marked as overused in the target-distribution guard report.
```

### 3.2 Causal-TOF Weighting

```bash
python scripts/run_causal_tof_weighting.py \
  --generated-pkl path/to/generated.pkl \
  --guarded-hints-json path/to/guarded_reweighted_gss_hints.json \
  --target-pkl path/to/target_normal.pkl \
  --out-scores outputs/gcad_gss_stage4/demo/causal_tof_scores.json \
  --out-weights outputs/gcad_gss_stage4/demo/generated.weights.json \
  --out-weighted-resampled-pkl outputs/gcad_gss_stage4/demo/generated_weighted_resampled.pkl \
  --mode weight \
  --temperature 2.0 \
  --seed 2024
```

### 3.3 Verifier + Repair Prompt

```bash
python scripts/verify_and_repair_generated_sequences.py \
  --generated-pkl path/to/generated.pkl \
  --guarded-hints-json path/to/guarded_reweighted_gss_hints.json \
  --guard-report-json path/to/guard_report.json \
  --target-pkl path/to/target_normal.pkl \
  --out-dir outputs/gcad_gss_stage4/demo/repair
```

该脚本不做 token-level decoding constraint，也不调用外部 API，只写 repair prompts。每条 repair prompt 包含原始序列、violated causal edges、overused devices、target distribution warning、最小修改要求、合法设备动作要求和原格式返回要求。

### 3.4 Stage 4 实验脚本

新增：

```text
scripts/run_stage4a_guarded_reweighted_gss_fr_st.py
scripts/run_stage4a_guarded_reweighted_gss_sp_st.py
scripts/run_stage4b_ad_guarded_reweighted_gss_fr_st.py
scripts/run_stage4b_ad_guarded_reweighted_gss_sp_st.py
scripts/run_stage4b_ad_causal_tof_weighted_fr_st.py
scripts/run_stage4b_ad_causal_tof_weighted_sp_st.py
```

Stage 4A 会生成 prompt、resolved prior、guard report、reweighted hints、config，并在提供 generated pkl 时输出 generation quality metrics。

Stage 4B 不伪造 AD 结果：如果没有真实 SmartGuard/SmartGen downstream AD metrics，必须显式传 `--dry-run`，脚本会写出 transparent placeholder。

### 3.5 Summary 脚本

新增：

```text
scripts/summarize_stage4_guarded_reweighted.py
scripts/summarize_stage4_causal_tof.py
```

输出：

```text
outputs/gcad_gss_stage4/stage4_guarded_reweighted_summary.csv
outputs/gcad_gss_stage4/stage4_guarded_reweighted_summary.md
outputs/gcad_gss_stage4/stage4_causal_tof_summary.csv
outputs/gcad_gss_stage4/stage4_causal_tof_summary.md
```

baseline 缺失时不会崩溃，会标注 `missing`。

## 4. 测试结果

本环境已运行全量测试：

```bash
cd /mnt/data/work/CausalSmartHome
PYTHONPATH=. pytest -q tests
```

结果：

```text
29 passed in 22.98s
```

新增测试覆盖：

- `resolve_gcad_prior` 能读取已有 `causal_prior.json`。
- 不传 prior 时调用已有 `GCADAdapter.mine_event_prior`，不是在 resolver 中写新算法。
- suppress guard 能把 overused endpoint 的 `guarded_weight` 设为 0。
- downweight guard 能按 factor 缩小权重。
- GCAD edge 能改变 GSS `final_score`。
- additive 与 multiplicative 模式结果不同。
- Causal-TOF 对 violated edge 给更高 `final_score` 和更低 `sample_weight`。
- repair prompt 包含 violated edge 和 overused device。
- JSON 输出可序列化。
- 新增脚本支持 `--help`。

## 5. Toy sanity run

由于本容器没有完整 FR/SP/US 数据集，也没有外部 API 配置，本次没有声称完成真实 SP-ST/FR-ST 论文级实验。已运行一个 toy sanity run，验证 Stage 4 接口、guard、reweight、Causal-TOF 与 repair prompt 能串起来。

输入输出位于：

```text
outputs/gcad_gss_stage4/demo_toy/
outputs/gcad_gss_stage4/demo_toy_stage4a_sp_wrapper/
outputs/gcad_gss_stage4/demo_toy_stage4b_sp_wrapper/
```

主要命令：

```bash
python scripts/build_guarded_causal_reweighted_gss_prompt.py \
  --source-pkl outputs/gcad_gss_stage4/demo_toy/input/source.pkl \
  --target-pkl outputs/gcad_gss_stage4/demo_toy/input/target.pkl \
  --prior-json outputs/gcad_gss_stage4/demo_toy/input/toy_causal_prior.json \
  --device-dict outputs/gcad_gss_stage4/demo_toy/input/device_dict.json \
  --out-prompt outputs/gcad_gss_stage4/demo_toy/prompt.txt \
  --out-prior-json outputs/gcad_gss_stage4/demo_toy/resolved_gcad_prior.json \
  --out-guard-report outputs/gcad_gss_stage4/demo_toy/guard_report.json \
  --out-reweighted-hints outputs/gcad_gss_stage4/demo_toy/guarded_reweighted_gss_hints.json \
  --lambda-causal 1.0 \
  --reweight-mode multiplicative \
  --guard-mode suppress \
  --max-overuse-ratio 1.25 \
  --top-k 20 \
  --seed 2024

python scripts/run_causal_tof_weighting.py \
  --generated-pkl outputs/gcad_gss_stage4/demo_toy/input/generated.pkl \
  --guarded-hints-json outputs/gcad_gss_stage4/demo_toy/guarded_reweighted_gss_hints.json \
  --target-pkl outputs/gcad_gss_stage4/demo_toy/input/target.pkl \
  --out-scores outputs/gcad_gss_stage4/demo_toy/causal_tof_scores.json \
  --out-weights outputs/gcad_gss_stage4/demo_toy/generated.weights.json \
  --out-weighted-resampled-pkl outputs/gcad_gss_stage4/demo_toy/generated_weighted_resampled.pkl \
  --mode weight \
  --temperature 2.0 \
  --seed 2024

python scripts/verify_and_repair_generated_sequences.py \
  --generated-pkl outputs/gcad_gss_stage4/demo_toy/input/generated.pkl \
  --guarded-hints-json outputs/gcad_gss_stage4/demo_toy/guarded_reweighted_gss_hints.json \
  --guard-report-json outputs/gcad_gss_stage4/demo_toy/guard_report.json \
  --target-pkl outputs/gcad_gss_stage4/demo_toy/input/target.pkl \
  --out-dir outputs/gcad_gss_stage4/demo_toy/repair
```

Toy Stage 4A wrapper 结果摘要：

```json
{
  "generated_size": 5,
  "target_size": 12,
  "low_evidence_rate": 0.0,
  "causal_coverage": 0.660869563865301,
  "causal_violation_rate": 0.33913043613469895,
  "action_js_to_target": 0.11281881344013167,
  "device_js_to_target": 0.3949097209597205,
  "transition_js_to_target": 0.7849784721044707,
  "tof_kept_rate": 1.0,
  "guarded_edge_count": 3,
  "suppressed_edge_count": 1,
  "downweighted_edge_count": 0,
  "avg_guarded_causal_strength": 0.2875000014901161
}
```

Toy guard report 中，`Light -> Television` 被 suppress：

```text
target endpoint Television overused: observed_freq=0.139535, target_freq=0.000000, ratio=139.535
```

这证明 guard report 能明确暴露类似 SP-ST `* -> Television` bias；真实 SP-ST 是否修复，需要真实 SP-ST source/target/generation 数据运行。

## 6. 负结果、tradeoff 和当前未完成项

当前未完成真实结论：

- FR-ST 是否提升：未在本容器跑真实 FR-ST Stage 4B AD，不能 claim。
- SP-ST 是否修复 Television bias：toy run 已验证机制，真实 SP-ST 仍需跑完整数据。
- guarded causal-reweighted GSS 是否优于 prompt-only guarded：需要把 Stage 4A/4B 与 Stage 3 guarded-edge prompt-only 同 seeds 对齐比较。
- Causal-TOF soft weighting 是否比 hard deletion 稳：当前只完成实现与 toy sanity，真实结论需 downstream AD。
- FPR tradeoff：必须看真实 `downstream_ad_metrics.json`。
- seed sensitivity：必须跑 seeds 2024/2025/2026。

可能 tradeoff：

1. suppress 太强可能丢掉目标上下文中仍然合理但低频的 causal edge。
2. downweight 比 suppress 更温和，但如果 raw prior bias 很强，可能仍诱导过生成。
3. multiplicative reweighting 保守地增强已有 GSS edge；additive 更容易把 causal signal 推高，可能提升结构一致性但增加 target distribution 偏差。
4. Causal-TOF soft weighting 保留样本多样性，但如果 downstream AD 不支持 sample weights，weighted resampling 仍可能引入复制偏差。

## 7. 下一步建议

1. 在服务器上用真实 SP-ST 跑：

```bash
python scripts/run_stage4a_guarded_reweighted_gss_sp_st.py \
  --source-pkl <sp_daytime_source_normal.pkl> \
  --target-pkl <sp_spring_target_normal.pkl> \
  --prior-json <stage3_or_new_sp_prior.json> \
  --device-dict <sp_dictionary.py_or_json> \
  --generated-pkl <smartgen_generated.pkl> \
  --out-dir outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024 \
  --seed 2024
```

2. 对 FR-ST 同样跑 seed 2024。
3. 扩展 seeds 2025/2026。
4. 比较 suppress vs downweight、additive vs multiplicative。
5. 对同一 generated pkl 运行 Causal-TOF：

```bash
python scripts/run_causal_tof_weighting.py \
  --generated-pkl <generated.pkl> \
  --guarded-hints-json outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024/guarded_reweighted_gss_hints.json \
  --target-pkl <sp_spring_target_normal.pkl> \
  --out-scores outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024/causal_tof_scores.json \
  --out-weights outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024/generated.weights.json \
  --out-weighted-resampled-pkl outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024/generated_weighted_resampled.pkl
```

6. 把 weighted/resampled synthetic pkl 接入 Stage 4B AD，保存真实 `downstream_ad_metrics.json`。
7. 运行 summary：

```bash
python scripts/summarize_stage4_guarded_reweighted.py
python scripts/summarize_stage4_causal_tof.py
```

## 8. 2026-06-22 server run 补记

在 `/home/heyang/projects/CausalSmartHome` 上完成了 Stage 4 服务器验收和真实 FR-ST/SP-ST Stage 4A glue run。外部项目均存在：

```text
/home/heyang/projects/SmartGen
/home/heyang/projects/SmartGuard
/home/heyang/projects/GCAD
```

`external_sources/SmartGen`、`external_sources/SmartGuard`、`external_sources/GCAD` 均可解析到上述真实项目路径。

真实输入路径：

| Scenario | Source normal pkl | Target normal pkl | Generated pkl | GCAD prior |
| --- | --- | --- | --- | --- |
| FR-ST seed 2024 | `/home/heyang/projects/SmartGen/behavior_prediciton_baseline/SASRec/baseline_data/fr/winter/trn.pkl` | `/home/heyang/projects/SmartGen/parameter_study/test/fr/spring/split_test.pkl` | `outputs/gcad_gss/fr_st_codex_calibrated_v3/fr_st_enhanced/smartgen_tof.pkl` | `outputs/gcad_gss/fr_st_codex_calibrated_v3/fr_st_enhanced/quality_eval/causal_prior_source.json` |
| FR-ST seed 2025 | same FR source | same FR target | `outputs/gcad_gss/fr_st_codex_calibrated_seed2025/fr_st_enhanced/smartgen_tof.pkl` | `outputs/gcad_gss/fr_st_codex_calibrated_seed2025/fr_st_enhanced/quality_eval/causal_prior_source.json` |
| FR-ST seed 2026 | same FR source | same FR target | `outputs/gcad_gss/fr_st_codex_calibrated_seed2026/fr_st_enhanced/smartgen_tof.pkl` | `outputs/gcad_gss/fr_st_codex_calibrated_seed2026/fr_st_enhanced/quality_eval/causal_prior_source.json` |
| SP-ST seed 2024 | `/home/heyang/projects/SmartGen/behavior_prediciton_baseline/SASRec/baseline_data/sp/daytime/trn.pkl` | `/home/heyang/projects/SmartGen/parameter_study/test/sp/spring/split_test.pkl` | `outputs/gcad_gss/sp_st_codex_calibrated_seed2024/sp_st_enhanced/smartgen_tof.pkl` | `outputs/gcad_gss/sp_st_codex_calibrated_seed2024/sp_st_enhanced/quality_eval/causal_prior_source.json` |
| SP-ST seed 2025 | same SP source | same SP target | `outputs/gcad_gss/sp_st_codex_calibrated_seed2025/sp_st_enhanced/smartgen_tof.pkl` | `outputs/gcad_gss/sp_st_codex_calibrated_seed2025/sp_st_enhanced/quality_eval/causal_prior_source.json` |
| SP-ST seed 2026 | same SP source | same SP target | `outputs/gcad_gss/sp_st_codex_calibrated_seed2026/sp_st_enhanced/smartgen_tof.pkl` | `outputs/gcad_gss/sp_st_codex_calibrated_seed2026/sp_st_enhanced/quality_eval/causal_prior_source.json` |

Primary suppress-mode outputs:

```text
outputs/gcad_gss_stage4/fr_st_guarded_reweighted_seed2024
outputs/gcad_gss_stage4/fr_st_guarded_reweighted_seed2025
outputs/gcad_gss_stage4/fr_st_guarded_reweighted_seed2026
outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024
outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2025
outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2026
```

Primary Stage 4A metrics:

| Scenario | Seeds | Mean generated | Mean action JS | Mean device JS | Mean transition JS | Suppressed edges | Guarded nonzero edges |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FR-ST suppress | 2024/2025/2026 | 208.0 | 0.6534 | 0.4596 | 0.8046 | 15 | 0 |
| SP-ST suppress | 2024/2025/2026 | 38.0 | 0.8219 | 0.6635 | 0.8731 | 48 | 0 |

SP-ST Television diagnosis is confirmed in the Stage 4 guard report. For seed 2024, examples in
`outputs/gcad_gss_stage4/sp_st_guarded_reweighted_seed2024/guard_report.json` show
`AirConditioner -> Television` suppressed with:

```text
target endpoint Television overused: observed_freq=0.668817, target_freq=0.031471, ratio=21.252
```

Important caveat: in the primary `guard-mode=suppress` runs, every top causal edge was suppressed
for both FR-ST and SP-ST. Thus the artifacts prove that GCAD prior is resolved, guarded, and wired
into the reweighted GSS builder, but the final nonzero causal contribution is zero after the default
guard. This is a conservative guard behavior, not evidence that causal reweighting itself improves
generation.

Extra seed-2024 downweight ablations were run to verify nonzero causal-reweight behavior:

| Output dir | Guard mode | Reweight mode | Nonzero guarded edges | Downweighted edges | Observation |
| --- | --- | --- | ---: | ---: | --- |
| `fr_st_downweight_multiplicative_seed2024` | downweight | multiplicative | 5 | 15 | Conservative; top rows remain dominated by transition score. |
| `fr_st_downweight_additive_seed2024` | downweight | additive | 9 | 15 | Additive can lift low-transition causal edges such as `AirPurifier -> Switch`. |
| `sp_st_downweight_multiplicative_seed2024` | downweight | multiplicative | 9 | 48 | Keeps weak `* -> Television`/other signals but reduces strength. |
| `sp_st_downweight_additive_seed2024` | downweight | additive | 9 | 48 | Additive lifts `AirConditioner -> Television` to top rank despite downweighting. |

Causal-TOF was run for the six primary suppress outputs. Because nonzero guarded causal edges were
zero, average causal violation is 0 and the soft weights are driven by target-distribution penalty,
not by causal-order repair. Mean sample weights:

| Scenario | Seed 2024 | Seed 2025 | Seed 2026 |
| --- | ---: | ---: | ---: |
| FR-ST | 0.2981 | 0.3055 | 0.3026 |
| SP-ST | 0.2244 | 0.2216 | 0.2227 |

Verifier/repair prompts were generated under each primary output's `repair/` directory. They include
violated causal edges, overused devices, and the target-distribution guard report. In suppress mode,
the violated-edge lists are empty because all causal edges were suppressed; the prompts are triggered
by distribution penalty.

Stage 4B downstream AD was not truly completed in this run. The scripts wrote dry-run
`downstream_ad_metrics.json` records under:

```text
outputs/gcad_gss_stage4/fr_st_stage4b_ad_guarded_reweighted_seed*
outputs/gcad_gss_stage4/sp_st_stage4b_ad_guarded_reweighted_seed*
outputs/gcad_gss_stage4/fr_st_stage4b_ad_causal_tof_weighted_seed*
outputs/gcad_gss_stage4/sp_st_stage4b_ad_causal_tof_weighted_seed*
```

No F1/precision/recall/FPR improvement should be claimed for Stage 4 until real SmartGuard or
SmartGen downstream AD metrics are produced from the Stage 4 generated or weighted/resampled pkl.
