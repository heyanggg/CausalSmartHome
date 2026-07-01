"""主实验路径推断工具。

实验级 main 入口只需要 dataset/scenario/seed/variant，就可以通过这些函数找到
对应阶段的输入输出位置。底层阶段脚本仍然保留显式路径参数，便于调试。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .gen_downstream_ad import ENV_BY_SCENARIO, SCENARIO_BY_ENV, SOURCE_ENV_BY_TARGET_ENV, env_for_scenario


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 规范化后的项目目录。旧结构已不再作为运行兜底，所有实验入口都从这里解析。
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "main_experiment"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "main_runs"
GEN_ROOT = PROJECT_ROOT / "causal_smart_home" / "gen_runtime"
DATA_ROOT = PROJECT_ROOT / "data" / "gen"
GEN_RUNTIME_DATA_ROOT = PROJECT_ROOT / "data" / "gen_runtime"
REFERENCE_ROOT = PROJECT_ROOT / "data" / "reference_gen"
DICTIONARY_PY = DATA_ROOT / "dictionary.py"


def normalize_project_path(path: str | Path) -> Path:
    """把相对路径解释为项目根目录下的绝对路径。"""
    raw = Path(path)
    return raw if raw.is_absolute() else PROJECT_ROOT / raw

ABLATION_VARIANT = "ablation_no_causal_tof"
PROPOSED_VARIANT = "proposed_causal_gss_codex_causal_tof"
VARIANTS = {ABLATION_VARIANT, PROPOSED_VARIANT}

# 主实验历史配置并不总是使用同一个 target-normal 文件：部分 spring/night
# 单元格用 split_test.pkl 来表示下游 AD 的 target-normal 集合，multiple 和 US
# 单元格则使用 test.pkl。这里把已接受主实验口径固定下来，供新 main 入口推断。
TARGET_PKL_NAME_BY_DATASET_ENV = {
    ("fr", "spring"): "split_test.pkl",
    ("fr", "night"): "split_test.pkl",
    ("sp", "spring"): "split_test.pkl",
}


@dataclass(frozen=True)
class StagePaths:
    """一个 dataset/scenario/seed 下常用阶段文件路径。"""

    root: Path
    dataset: str
    scenario: str
    seed: int

    @property
    def key(self) -> str:
        return experiment_key(self.dataset, self.scenario)

    @property
    def seed_dir(self) -> Path:
        return self.root / self.key / f"seed{self.seed}"

    @property
    def pre_tof_pkl(self) -> Path:
        return self.seed_dir / "codex_generation" / "generated_codex.pkl"

    @property
    def gen_tof_pkl(self) -> Path:
        return self.seed_dir / "gen_original_tof" / "gen_tof.pkl"

    @property
    def causal_tof_pkl(self) -> Path:
        return self.seed_dir / "causal_tof" / "generated_gen_tof_causal_tof.pkl"

    @property
    def guarded_hints_json(self) -> Path:
        return self.seed_dir / "causal_gss" / "guarded_reweighted_gss_hints.json"

    @property
    def causal_gss_dir(self) -> Path:
        return self.seed_dir / "causal_gss"

    @property
    def causal_gss_config(self) -> Path:
        return self.causal_gss_dir / "config.json"

    @property
    def generation_package_dir(self) -> Path:
        return self.seed_dir / "codex_generation_package"

    @property
    def codex_generation_dir(self) -> Path:
        return self.seed_dir / "codex_generation"


def experiment_key(dataset: str, scenario: str) -> str:
    """返回目录中使用的实验 key，例如 ``us_st``。"""
    return f"{dataset}_{scenario_key(scenario)}"


def scenario_key(scenario: str) -> str:
    """把 ``spring/night/multiple`` 或 ``st/tt/nt`` 统一成目录短名。"""
    return SCENARIO_BY_ENV[env_for_scenario(scenario)]


def stage_paths(root: str | Path, dataset: str, scenario: str, seed: int) -> StagePaths:
    """构造某个实验单元的常用阶段路径集合。"""
    if scenario not in ENV_BY_SCENARIO:
        valid = ", ".join(sorted(ENV_BY_SCENARIO))
        raise ValueError(f"scenario must be one of: {valid}")
    return StagePaths(root=Path(root), dataset=dataset, scenario=scenario, seed=int(seed))


def source_pkl_for(dataset: str, scenario: str) -> Path:
    """返回该目标场景对应的源上下文 normal training pkl。"""
    env = env_for_scenario(scenario)
    source_env = SOURCE_ENV_BY_TARGET_ENV[env]
    return DATA_ROOT / dataset / source_env / "trn.pkl"


def target_pkl_for(dataset: str, scenario: str) -> Path:
    """返回该目标场景的 target normal pkl，用于 guard 或 Causal-TOF 分布项。"""
    env = env_for_scenario(scenario)
    filename = TARGET_PKL_NAME_BY_DATASET_ENV.get((dataset, env), "test.pkl")
    return DATA_ROOT / dataset / env / filename


def target_pkl_from_stage_config(paths: StagePaths) -> Path | None:
    """从已有 causal-GSS 配置中读取 target pkl，缺失时返回 None。

    这用于复现实验阶段运行：已有 cell 的 target 选择以当时保存的配置为准，
    新 cell 才落回 ``target_pkl_for`` 的主实验默认映射。
    """
    if not paths.causal_gss_config.exists():
        return None
    try:
        payload = json.loads(paths.causal_gss_config.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    target = payload.get("target_pkl")
    if not target:
        return None
    return normalize_project_path(target)


def default_downstream_out_dir(root: str | Path, dataset: str, scenario: str, seed: int, variant: str) -> Path:
    """返回实验级 main 默认写下游 AD 结果的位置。"""
    return Path(root) / experiment_key(dataset, scenario) / f"seed{seed}" / "downstream_ad" / variant


def default_causal_tof_dir(root: str | Path, dataset: str, scenario: str, seed: int) -> Path:
    """返回实验级 main 默认写 Causal-TOF 结果的位置。"""
    return Path(root) / experiment_key(dataset, scenario) / f"seed{seed}" / "causal_tof"


def input_for_variant(paths: StagePaths, variant: str) -> tuple[Path, Path | None, Path | None]:
    """根据 variant 推断下游 AD 的 generated/pre_tof/gen_tof 参数。"""
    if variant == ABLATION_VARIANT:
        return paths.gen_tof_pkl, paths.pre_tof_pkl if paths.pre_tof_pkl.exists() else None, None
    if variant == PROPOSED_VARIANT:
        return (
            paths.causal_tof_pkl,
            paths.pre_tof_pkl if paths.pre_tof_pkl.exists() else None,
            paths.gen_tof_pkl,
        )
    raise ValueError(f"unknown variant: {variant}")


def require_file(path: str | Path, label: str) -> Path:
    """返回存在的文件路径；不存在时给出清晰错误。"""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved
