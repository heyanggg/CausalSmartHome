from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DATASETS = ["fr", "sp", "us"]

SCENARIOS: dict[str, dict[str, str]] = {
    "st": {
        "name": "seasonal transition",
        "source_context": "winter",
        "target_context": "spring",
    },
    "tt": {
        "name": "time-schedule transition",
        "source_context": "daytime",
        "target_context": "nighttime",
    },
    "nt": {
        "name": "occupancy-number transition",
        "source_context": "single",
        "target_context": "multiple",
    },
}

SEEDS = [2024, 2025, 2026]

GEN_CONTEXT_ALIASES: dict[str, tuple[str, ...]] = {
    "winter": ("winter",),
    "spring": ("spring",),
    "daytime": ("daytime", "day"),
    "nighttime": ("nighttime", "night"),
    "single": ("single",),
    "multiple": ("multiple", "multi"),
}

VOCAB_SIZES = {"fr": 223, "sp": 235, "us": 269}

PROPOSED_VARIANT = "proposed_causal_gss_gpt55_causal_tof"
ABLATION_VARIANT = "ablation_no_causal_tof"
REFERENCE_VARIANT = "original_gen_reference"


@dataclass(frozen=True)
class MatrixItem:
    dataset: str
    scenario: str
    name: str
    source_context: str
    target_context: str
    source_aliases: tuple[str, ...]
    target_aliases: tuple[str, ...]

    @property
    def key(self) -> str:
        return scenario_key(self.dataset, self.scenario)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"key": self.key}


def scenario_key(dataset: str, scenario: str) -> str:
    return f"{dataset}_{scenario}"


def validate_dataset(dataset: str) -> str:
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}: {dataset}")
    return dataset


def validate_scenario(scenario: str) -> str:
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {list(SCENARIOS)}: {scenario}")
    return scenario


def matrix_item(dataset: str, scenario: str) -> MatrixItem:
    dataset = validate_dataset(dataset)
    scenario = validate_scenario(scenario)
    spec = SCENARIOS[scenario]
    source = spec["source_context"]
    target = spec["target_context"]
    return MatrixItem(
        dataset=dataset,
        scenario=scenario,
        name=spec["name"],
        source_context=source,
        target_context=target,
        source_aliases=GEN_CONTEXT_ALIASES[source],
        target_aliases=GEN_CONTEXT_ALIASES[target],
    )


def experiment_grid() -> list[MatrixItem]:
    return [matrix_item(dataset, scenario) for dataset in DATASETS for scenario in SCENARIOS]


EXPERIMENT_GRID = experiment_grid()


def iter_selected(matrix: str = "all", dataset: str | None = None, scenario: str | None = None) -> Iterable[MatrixItem]:
    if matrix == "all":
        yield from EXPERIMENT_GRID
        return
    if dataset is None or scenario is None:
        raise ValueError("--dataset and --scenario are required when --matrix is not all")
    yield matrix_item(dataset, scenario)


def resolve_existing_context(root: Path, dataset: str, context: str) -> str:
    for alias in GEN_CONTEXT_ALIASES.get(context, (context,)):
        if (root / dataset / alias).exists():
            return alias
    return GEN_CONTEXT_ALIASES.get(context, (context,))[0]


def target_env_for_scenario(scenario: str, root: Path | None = None, dataset: str | None = None) -> str:
    item = matrix_item(dataset or DATASETS[0], scenario)
    if root is not None and dataset is not None:
        return resolve_existing_context(root, dataset, item.target_context)
    return item.target_aliases[0]


def source_env_for_scenario(scenario: str, root: Path | None = None, dataset: str | None = None) -> str:
    item = matrix_item(dataset or DATASETS[0], scenario)
    if root is not None and dataset is not None:
        return resolve_existing_context(root, dataset, item.source_context)
    return item.source_aliases[0]


def legacy_stage_dir(root: Path, stage: str, dataset: str, scenario: str, seed: int) -> Path:
    return root / stage / scenario_key(dataset, scenario) / f"seed{seed}"


def matrix_stage_dir(root: Path, stage: str, dataset: str, scenario: str, seed: int) -> Path:
    return root / stage / dataset / scenario / f"seed{seed}"


def existing_stage_dir(root: Path, stage: str, dataset: str, scenario: str, seed: int) -> Path:
    matrix_dir = matrix_stage_dir(root, stage, dataset, scenario, seed)
    legacy_dir = legacy_stage_dir(root, stage, dataset, scenario, seed)
    if matrix_dir.exists():
        return matrix_dir
    return legacy_dir


def resource_paths(resources_root: Path, dataset: str, scenario: str) -> dict[str, Path]:
    item = matrix_item(dataset, scenario)
    source_env = resolve_existing_context(resources_root, dataset, item.source_context)
    target_env = resolve_existing_context(resources_root, dataset, item.target_context)
    return {
        "source_pkl": resources_root / dataset / source_env / "trn.pkl",
        "target_split_pkl": resources_root / dataset / target_env / "split_test.pkl",
        "target_test_pkl": resources_root / dataset / target_env / "test.pkl",
        "target_train_pkl": resources_root / dataset / target_env / "trn.pkl",
        "target_validation_pkl": resources_root / dataset / target_env / "vld.pkl",
    }


def missing_paths(paths: dict[str, Path]) -> list[str]:
    return [f"{name}: {path}" for name, path in paths.items() if not path.exists()]


def load_reference_baseline(path: Path | None = None) -> dict[tuple[str, str], dict[str, Any]]:
    ref_path = path or Path(__file__).resolve().parent / "resources" / "reference" / "smartgen_table3_ad.json"
    payload = json.loads(ref_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", payload)
    return {(str(row["dataset"]), str(row["scenario"])): row for row in rows}
