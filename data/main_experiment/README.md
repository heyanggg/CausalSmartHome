# Main Experiment Artifacts

本目录保存 2026-07-11 之前 target-reference-assisted 方法的历史阶段资料、配置、
指标和必要的中间 pkl。它们不属于当前 Zero-target-data 正式方法。

```text
data/main_experiment/{dataset}_{scenario}/seed{seed}/
  causal_gss/
  codex_generation/
  codex_generation_package/
  gen_original_tof/
  causal_tof/
  downstream_ad/
```

当前正式新实验不要以这里的 target distribution、Guard 或 Causal-TOF 产物作为输入。
新实验写入 `outputs/zero_target_runs/`：

```bash
python scripts/main_prepare_generation.py --dataset fr --scenario tt --seed 2024 --out-root outputs/zero_target_runs
python scripts/main_run_downstream_ad.py --dataset fr --scenario tt --seed 2024 --variant proposed_zero_target_causal_gss_codex --input-root outputs/zero_target_runs --out-root outputs/zero_target_runs
```

本目录中的大型 pkl 和运行输出不提交到 Git；需要完整复现实验时，应在本地保留或同步
对应的 `data/main_experiment/...` 文件树。

检修或搬迁项目后，用统一健康检查确认正式结果的 JSON、seed 坐标和运行资产没有损坏：

```bash
python scripts/check_project.py --runs-root data/main_experiment
```

新复现结果应写入 `outputs/zero_target_runs/`，不要直接覆盖这里的历史结果。例如只运行
proposed（不运行消融）可执行：

```bash
python scripts/main_run_downstream_ad.py \
  --dataset fr --scenario tt --seed 2024 \
  --variant proposed_zero_target_causal_gss_codex \
  --input-root outputs/zero_target_runs --out-root outputs/zero_target_runs \
  --device cuda --cuda-visible-devices 0
```
