"""Gen 原始两阶段 TOF 的包装器。

本项目刻意不重写 SmartGen/Gen 的 TOF 逻辑。这个包装器只负责把生成数据复制到
项目内 Gen 运行代码期望的目录结构中，调用 ``security_check.py``，然后把 Gen
选择出的结果复制回 CausalSmartHome 的运行目录，并写出审计报告。
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gen_downstream_ad import DEFAULT_THRESHOLDS, env_for_scenario, load_pickle


@dataclass(frozen=True)
class GenOriginalTOFConfig:
    """调用 Gen 原始 two-stage TOF 所需的配置。

    包装器不重新实现 TOF。真实运行时，它会把 fresh generated pkl 放进 Gen 的
    固定路径布局，然后调用
    ``security_check.security_check(dataset, env, threshold, method, model)``。
    原始脚本内部先做 reconstruction-loss outlier detection，再做 utility/value
    selection。
    """

    gen_root: Path
    dataset: str
    scenario: str
    generated_pkl: Path
    out_pkl: Path
    out_dir: Path
    seed: int = 2024
    method: str = "SPPC"
    model: str = "gpt-4o"
    threshold: str | None = None
    security_check_path: Path | None = None
    python_executable: str = sys.executable
    cuda_visible_devices: str | None = "0"
    dry_run: bool = False


def gen_env_for_scenario(scenario: str) -> str:
    return env_for_scenario(scenario)


def gen_code_dir(gen_root: str | Path) -> Path:
    """定位包含 Gen ``security_check.py`` 的代码目录。"""
    root = Path(gen_root).resolve()
    candidates = [root / "gen_original_tof", root]
    for candidate in candidates:
        if (candidate / "security_check.py").exists():
            return candidate
    # 如果没找到，返回最可能的路径，让后续错误信息能直接指出缺失文件位置。
    return root / "gen_original_tof"


def resolve_security_check_path(config: GenOriginalTOFConfig) -> Path:
    if config.security_check_path is not None:
        path = config.security_check_path.resolve()
    else:
        path = gen_code_dir(config.gen_root) / "security_check.py"
    if not path.exists():
        raise FileNotFoundError(
            "Gen original security_check.py was not found. "
            f"Looked for: {path}. Set --gen-root or --security-check-path to the bundled Gen TOF code."
        )
    return path


def expected_gen_tof_paths(code_dir: Path, dataset: str, env: str, threshold: str, method: str, model: str) -> dict[str, Path]:
    """复原 Gen TOF 原脚本硬编码使用的输入/输出文件名。"""
    base = f"{dataset}_{env}_generation_{method}_th={threshold}_{model}_seq"
    filter_dir = code_dir / "filter_data" / dataset / env
    return {
        "input": filter_dir / f"{base}.pkl",
        "filter_output": filter_dir / f"{base}_filter.pkl",
        "utility_selected": filter_dir / f"{base}_filter_true.pkl",
        "filter_train": filter_dir / f"{base}_filter_trn.pkl",
        "filter_validation": filter_dir / f"{base}_filter_vld.pkl",
        "outlier_prefix": filter_dir / f"{base}_filter_out",
        "check_model": code_dir / "check_model" / f"best_{dataset}_{model}_{method}.pth",
    }


def count_pickle_items(path: str | Path) -> int | None:
    try:
        obj = load_pickle(path)
    except Exception:
        return None
    try:
        return len(obj)
    except Exception:
        return None


def run_gen_original_tof(config: GenOriginalTOFConfig) -> dict[str, Any]:
    """执行 Gen TOF、收集溯源信息，并返回机器可读报告。"""
    env = gen_env_for_scenario(config.scenario)
    threshold = config.threshold or DEFAULT_THRESHOLDS[(config.dataset, env)]
    out_dir = config.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pkl = config.out_pkl.resolve()
    out_pkl.parent.mkdir(parents=True, exist_ok=True)

    input_count = count_pickle_items(config.generated_pkl)
    report: dict[str, Any] = {
        "dataset": config.dataset,
        "scenario": config.scenario,
        "gen_env": env,
        "seed": config.seed,
        "method": config.method,
        "model": config.model,
        "threshold": threshold,
        "input_pkl": str(config.generated_pkl.resolve()),
        "out_pkl": str(out_pkl),
        "num_generated_before_tof": input_count,
        "used_gen_original_tof": False,
        "gen_original_tof_filter": "reconstruction_loss_iqr_outlier_detection",
        "gen_original_tof_utility_selection": "utility_value_selection",
        "dry_run": bool(config.dry_run),
    }

    if config.dry_run:
        shutil.copyfile(config.generated_pkl.resolve(), out_pkl)
        report.update(
            {
                "status": "dry_run_copied_input",
                "used_gen_original_tof": False,
                "num_generated_after_gen_tof": count_pickle_items(out_pkl),
                "note": "Dry run did not execute Gen security_check.py.",
            }
        )
        write_report(out_dir, report)
        return report

    security_check = resolve_security_check_path(config)
    code_dir = security_check.parent.resolve()
    paths = expected_gen_tof_paths(code_dir, config.dataset, env, threshold, config.method, config.model)
    paths["input"].parent.mkdir(parents=True, exist_ok=True)
    paths["check_model"].parent.mkdir(parents=True, exist_ok=True)
    # Gen 原脚本只会从自己的 filter_data 路径读文件，因此先把生成 pkl 暂存到
    # 那个固定位置，再调用原始 security_check 函数。
    shutil.copyfile(config.generated_pkl.resolve(), paths["input"])

    env_vars = os.environ.copy()
    if config.cuda_visible_devices is not None:
        env_vars["CUDA_VISIBLE_DEVICES"] = str(config.cuda_visible_devices)
    env_vars["PYTHONPATH"] = str(code_dir) + os.pathsep + env_vars.get("PYTHONPATH", "")

    code = (
        "import importlib.util, os; "
        f"os.chdir({str(code_dir)!r}); "
        f"spec=importlib.util.spec_from_file_location('gen_security_check', {str(security_check)!r}); "
        "mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); "
        f"mod.setup_seed({int(config.seed)}) if hasattr(mod, 'setup_seed') else None; "
        f"mod.security_check({config.dataset!r}, {env!r}, {threshold!r}, {config.method!r}, {config.model!r})"
    )
    cmd = [config.python_executable, "-c", code]
    report["security_check_path"] = str(security_check)
    report["gen_code_dir"] = str(code_dir)
    report["command"] = " ".join(shlex.quote(part) for part in cmd)
    report["expected_paths"] = {key: str(value) for key, value in paths.items()}

    completed = subprocess.run(cmd, cwd=str(config.gen_root.resolve()), env=env_vars, text=True, capture_output=True)
    report["returncode"] = completed.returncode
    report["stdout_tail"] = completed.stdout[-4000:]
    report["stderr_tail"] = completed.stderr[-4000:]
    if completed.returncode != 0:
        report.update({"status": "failed", "num_generated_after_gen_tof": None})
        write_report(out_dir, report)
        raise RuntimeError(
            "Gen original TOF failed; see gen_original_tof_report.json. "
            f"stderr tail: {completed.stderr[-1000:]}"
        )

    selected_source = None
    if paths["utility_selected"].exists():
        # Gen 两阶段 TOF 的优先最终产物：已经通过过滤并完成效用选择。
        selected_source = paths["utility_selected"]
        output_stage = "utility_selected"
    elif paths["filter_output"].exists():
        selected_source = paths["filter_output"]
        output_stage = "filter_output_no_utility_selection_file"
    else:
        selected_source = paths["input"]
        output_stage = "input_no_filter_file_found"
    shutil.copyfile(selected_source, out_pkl)

    report.update(
        {
            "status": "success",
            "used_gen_original_tof": True,
            "selected_source": str(selected_source),
            "output_stage": output_stage,
            "num_generated_after_gen_tof": count_pickle_items(out_pkl),
        }
    )
    write_report(out_dir, report)
    return report


def write_report(out_dir: Path, report: dict[str, Any]) -> None:
    (out_dir / "gen_original_tof_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Gen Original TOF Report",
        "",
        "| field | value |",
        "| --- | --- |",
    ]
    for key in [
        "status",
        "dataset",
        "scenario",
        "seed",
        "input_pkl",
        "out_pkl",
        "used_gen_original_tof",
        "num_generated_before_tof",
        "num_generated_after_gen_tof",
        "output_stage",
        "security_check_path",
    ]:
        lines.append(f"| {key} | {report.get(key, '')} |")
    (out_dir / "gen_original_tof_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
