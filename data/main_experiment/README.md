# Main Experiment Artifacts

本目录保存正式主实验的阶段资料、配置、指标和必要的中间 pkl。目录结构为：

```text
data/main_experiment/{dataset}_{scenario}/seed{seed}/
  causal_gss/
  codex_generation/
  codex_generation_package/
  gen_original_tof/
  causal_tof/
  downstream_ad/
```

实验入口默认从这里读取已经确定的主实验阶段输入：

```bash
python scripts/main_run_causal_tof_and_ad.py --dataset sp --scenario nt --seed 2024
python scripts/main_run_downstream_ad.py --dataset sp --scenario nt --seed 2024 --variant proposed_causal_gss_codex_causal_tof
```

本目录中的大型 pkl 和运行输出不提交到 Git；需要完整复现实验时，应在本地保留或同步
对应的 `data/main_experiment/...` 文件树。

检修或搬迁项目后，用统一健康检查确认正式结果的 JSON、seed 坐标和运行资产没有损坏：

```bash
python scripts/check_project.py --runs-root data/main_experiment
```

新复现结果应写入 `outputs/main_runs/`，不要直接覆盖这里的正式历史结果。例如只复现
proposed（不运行消融）可执行：

```bash
python scripts/main_run_downstream_ad.py \
  --dataset fr --scenario st --seed 2024 \
  --variant proposed_causal_gss_codex_causal_tof \
  --device cuda --cuda-visible-devices 0
```
