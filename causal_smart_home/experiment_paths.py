"""主实验路径推断工具。

实验级 main 入口只需要 dataset/scenario/seed/variant，就可以通过这些函数找到
对应阶段的输入输出位置。底层阶段脚本仍然保留显式路径参数，便于调试。
"""

from __future__ import annotations

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


PROPOSED_VARIANT = "proposed_zero_target_causal_gss_codex"
VARIANTS = {PROPOSED_VARIANT}


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
    def causal_reweighted_hints_json(self) -> Path:
        return self.seed_dir / "causal_gss" / "causal_reweighted_gss_hints.json"

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


def default_downstream_out_dir(root: str | Path, dataset: str, scenario: str, seed: int, variant: str) -> Path:
    """返回实验级 main 默认写下游 AD 结果的位置。"""
    return Path(root) / experiment_key(dataset, scenario) / f"seed{seed}" / "downstream_ad" / variant


def input_for_variant(paths: StagePaths, variant: str) -> tuple[Path, Path | None, Path | None]:
    """根据 variant 推断下游 AD 的 generated/pre_tof/gen_tof 参数。"""
    if variant == PROPOSED_VARIANT:
        return paths.gen_tof_pkl, paths.pre_tof_pkl if paths.pre_tof_pkl.exists() else None, paths.gen_tof_pkl
    raise ValueError(f"unknown variant: {variant}")


def require_file(path: str | Path, label: str) -> Path:
    """返回存在的文件路径；不存在时给出清晰错误。"""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved
