# Codex Handoff Prompt

你是新的 Codex coding agent，请继续维护 `/home/heyang/projects/CausalSmartHome` 项目。请用中文和用户沟通，默认主动执行，不要停留在建议层面。

项目背景：

- 项目根目录：`/home/heyang/projects/CausalSmartHome`
- 远端：`git@github.com:heyanggg/CausalSmartHome.git`
- 这是一个非侵入式 glue-layer 项目，用 wrapper / adapter 连接 SmartGen、SmartGuard 和 GCAD-style causal prior。
- 不要新建项目。
- 不要直接修改 SmartGen / SmartGuard / GCAD 原始源码主体。
- 允许修改 CausalSmartHome 自身的 wrapper、adapter、scripts、tests、docs。
- 大型实验输出在 `outputs/`，通常被 `.gitignore` 忽略，不要随便删除。
- 推荐环境：`/home/heyang/miniconda3/envs/smartguard_env/bin/python`
- GPU 服务器：优先 `--device cuda --cuda-visible-devices 0`。

当前主线：

- GCAD-GSS prompt enhancement wrapper。
- 目标是把 device-level GCAD prior 从 TOF 后处理前移到 SmartGen GSS / prompt 阶段，让因果先验参与“怎么生成”，而不是生成后复制、删样本或加权样本。
- 当前只做 FR winter -> spring，即 FR-ST。
- 暂时不要接 SmartGuard 作为主实验。

已新增关键代码：

- `causal_smart_home/causal_gss.py`
- `causal_smart_home/causal_prompt_adapter.py`
- `scripts/build_gcad_gss_prompt.py`
- `scripts/run_stage3a_gcad_gss_fr_st.py`
- `scripts/run_stage3b_ad_gcad_gss_fr_st.py`
- `scripts/evaluate_smartgen_gcad_quality.py`
- `scripts/summarize_stage3a_repeats.py`
- `scripts/summarize_stage3a_threshold_ablation.py`
- `tests/test_gcad_gss_prompt.py`
- `tests/test_evaluate_smartgen_gcad_quality.py`

最新文档：

- `README.md`
- `docs/task12_gcad_gss_prompt_stage3.md`

Stage 3A 最新可复现实验：

```bash
/home/heyang/miniconda3/envs/smartguard_env/bin/python scripts/run_stage3a_gcad_gss_fr_st.py \
  --stage all \
  --groups original,enhanced \
  --offline-generator codex-calibrated \
  --samples-per-category 6 \
  --no-reuse-existing-original \
  --output-tag fr_st_codex_calibrated_v3 \
  --sparse-threshold 0.001 \
  --epochs 80 \
  --sample-limit 64 \
  --seed 2024 \
  --require-cuda
```

Stage 3A 输出：

- `outputs/gcad_gss/fr_st_codex_calibrated_v3/fr_st_stage3a_summary.csv`
- `outputs/gcad_gss/fr_st_codex_calibrated_v3/fr_st_stage3a_summary.md`
- `outputs/gcad_gss/fr_st_codex_calibrated_v3/fr_st_original/smartgen_tof.pkl`
- `outputs/gcad_gss/fr_st_codex_calibrated_v3/fr_st_enhanced/smartgen_tof.pkl`

Stage 3A 结果：

- original: raw 222, TOF 207, coverage 0.5532, violation 0.0217, low evidence 0.4251, action JS 0.4784, device JS 0.2457, transition JS 0.7891
- enhanced: raw 222, TOF 209, coverage 0.5221, violation 0.0760, low evidence 0.4019, action JS 0.4234, device JS 0.2132, transition JS 0.8027
- enhanced prompt 改善 low evidence、action JS、device JS、TOF kept rate；但 causal coverage 降低、violation rate 上升、transition JS 略差。

Stage 3B downstream AD sanity check：

```bash
/home/heyang/miniconda3/envs/smartguard_env/bin/python scripts/run_stage3b_ad_gcad_gss_fr_st.py \
  --stage3a-tag fr_st_codex_calibrated_v3 \
  --epochs 15 \
  --seed 2024 \
  --device cuda \
  --cuda-visible-devices 0
```

Stage 3B 输出：

- `outputs/gcad_gss/fr_st_stage3b_ad/fr_st_codex_calibrated_v3/metrics.csv`
- `outputs/gcad_gss/fr_st_stage3b_ad/fr_st_codex_calibrated_v3/metrics.md`

Stage 3B 结果：

- original_prompt: precision 0.6241, recall 1.0000, F1 0.7686, FPR 0.6023, accuracy 0.6989
- enhanced_prompt: precision 0.7521, recall 1.0000, F1 0.8585, FPR 0.3295, accuracy 0.8352
- enhanced - original: F1 +0.0900, FPR -0.2727, accuracy +0.1364

重要 caveat：

- `codex-calibrated` 是本地可复现文本生成模式。
- 它使用现有 SmartGen FR-ST TOF baseline 作为 style bank，用于校准长度、设备/动作多样性和 transition 风格。
- 它仍然输出 SmartGen `<seq ... seq>` 文本，再经过 SmartGen 原 Extract / Transnum / security_check / TOF。
- 这支持工程 sanity check，但不能被包装成完全独立 LLM/API 生成的最终论文结论。

推荐谨慎表述：

```text
On FR-ST, under a controlled SmartGen wrapper where the prompt is the only intended intervention, the GCAD-GSS enhanced prompt improves several generation-quality indicators and yields a stronger downstream SmartGen Transformer Autoencoder AD sanity-check result than the original prompt.
```

工作习惯要求：

- 改文件前先读上下文。
- 使用 `rg` / `rg --files`。
- 用 `apply_patch` 做手工编辑。
- 不要删除用户实验输出，除非用户明确要求。
- 不要声称下游 AD 已经全面提升。
- 每次跑实验后记录命令、路径、指标和 caveat。

下一步任务：

1. 做 FR-ST `codex-calibrated` 多 seed 重复实验，至少 seeds `2024, 2025, 2026`，固定其他设置。
2. 汇总 Stage 3A 多 seed 均值/方差，并补跑对应 Stage 3B AD sanity check。
3. 判断 enhanced prompt 的提升是否稳定，重点看 F1、FPR、low evidence、action/device JS。
4. 如果多 seed 稳定，再准备真实 LLM/API Stage 3A 版本；仍保持唯一变量为 prompt。
