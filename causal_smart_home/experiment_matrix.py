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

STATUS_COMPLETE = "COMPLETE"
STATUS_DOWNSTREAM_READY = "DOWNSTREAM_READY"
STATUS_GENERATION_MISSING = "GENERATION_MISSING"
STATUS_DATA_READY = "DATA_READY"
STATUS_MISSING_DATA = "MISSING_DATA"
STATUS_MISSING_ATTACK = "MISSING_ATTACK"
STATUS_MISSING_CHECKPOINT = "MISSING_CHECKPOINT"

TARGET_ATTACK_SUFFIX = {
    "spring": "spring_attack_heater",
    "nighttime": "night_attack_time",
    "multiple": "multiple_attack_tv",
}

DEFAULT_THRESHOLDS_BY_SCENARIO = {
    ("fr", "st"): "0.918",
    ("fr", "tt"): "0.92",
    ("fr", "nt"): "0.915",
    ("sp", "st"): "0.915",
    ("sp", "tt"): "0.917",
    ("sp", "nt"): "0.915",
    ("us", "st"): "0.905",
    ("us", "tt"): "0.919",
    ("us", "nt"): "0.913",
}

DEFAULT_AD_PERCENTAGES_BY_SCENARIO = {
    ("fr", "st"): 95.5,
    ("fr", "tt"): 95.0,
    ("fr", "nt"): 99.0,
    ("sp", "st"): 95.0,
    ("sp", "tt"): 95.0,
    ("sp", "nt"): 99.0,
    ("us", "st"): 95.0,
    ("us", "tt"): 93.0,
    ("us", "nt"): 99.0,
}


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
        payload = asdict(self)
        payload["key"] = self.key
        return payload


@dataclass(frozen=True)
class MatrixCellPaths:
    dataset: str
    scenario: str
    source_context: str
    target_context: str
    source_train_pkl: Path
    target_train_pkl: Path
    target_validation_pkl: Path
    target_test_pkl: Path
    target_split_test_pkl: Path
    downstream_test_pkl: Path
    downstream_test_dir: Path
    downstream_attack_pkl: Path
    downstream_attack_dir: Path
    gen_original_tof_checkpoint: Path
    downstream_checkpoint: Path
    dictionary_py: Path

    def to_dict(self) -> dict[str, Any]:
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}


@dataclass(frozen=True)
class MatrixCellStatus:
    dataset: str
    scenario: str
    status: str
    missing_data: tuple[str, ...]
    missing_attack: tuple[str, ...]
    missing_checkpoint: tuple[str, ...]
    missing_generation: tuple[str, ...]
    missing_downstream: tuple[str, ...]
    paths: MatrixCellPaths

    @property
    def missing(self) -> tuple[str, ...]:
        return self.missing_data + self.missing_attack + self.missing_checkpoint + self.missing_generation + self.missing_downstream

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "scenario": self.scenario,
            "status": self.status,
            "missing_data": list(self.missing_data),
            "missing_attack": list(self.missing_attack),
            "missing_checkpoint": list(self.missing_checkpoint),
            "missing_generation": list(self.missing_generation),
            "missing_downstream": list(self.missing_downstream),
            "missing": list(self.missing),
            "paths": self.paths.to_dict(),
        }


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


def canonical_context_dir(context: str) -> str:
    if context == "nighttime":
        return "nighttime"
    return context


def attack_suffix_for_target_context(target_context: str) -> str:
    return TARGET_ATTACK_SUFFIX[target_context]


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


def resolve_matrix_cell_paths(dataset: str, scenario: str, root: Path) -> MatrixCellPaths:
    item = matrix_item(dataset, scenario)
    resources_root = root / "causal_smart_home" / "resources" / "gen_data"
    gen_core = root / "causal_smart_home" / "gen_core"
    source_dir = resources_root / dataset / canonical_context_dir(item.source_context)
    target_dir = resources_root / dataset / canonical_context_dir(item.target_context)
    target_env = canonical_context_dir(item.target_context)
    attack_suffix = attack_suffix_for_target_context(item.target_context)
    return MatrixCellPaths(
        dataset=dataset,
        scenario=scenario,
        source_context=item.source_context,
        target_context=item.target_context,
        source_train_pkl=source_dir / "trn.pkl",
        target_train_pkl=target_dir / "trn.pkl",
        target_validation_pkl=target_dir / "vld.pkl",
        target_test_pkl=target_dir / "test.pkl",
        target_split_test_pkl=target_dir / "split_test.pkl",
        downstream_test_pkl=gen_core / "anomaly_detection_pipeline" / "test" / dataset / target_env / "test.pkl",
        downstream_test_dir=gen_core / "anomaly_detection_pipeline" / "test" / dataset / target_env,
        downstream_attack_pkl=gen_core / "anomaly_detection_pipeline" / "attack" / dataset / f"labeled_{dataset}_{attack_suffix}.pkl",
        downstream_attack_dir=gen_core / "anomaly_detection_pipeline" / "attack" / dataset,
        gen_original_tof_checkpoint=gen_core / "gen_original_tof" / "check_model" / f"best_{dataset}_gpt-4o_SPPC.pth",
        downstream_checkpoint=gen_core / "anomaly_detection_pipeline" / "check_model" / f"best_{dataset}_gpt-4o_SPPC.pth",
        dictionary_py=resources_root / "dictionary.py",
    )


def check_matrix_cell_data_ready(dataset: str, scenario: str, root: Path) -> MatrixCellStatus:
    paths = resolve_matrix_cell_paths(dataset, scenario, root)
    data_requirements = {
        "source_train_pkl": paths.source_train_pkl,
        "target_test_pkl": paths.target_test_pkl,
        "target_split_test_pkl": paths.target_split_test_pkl,
        "dictionary_py": paths.dictionary_py,
    }
    attack_requirements = {
        "downstream_test_pkl": paths.downstream_test_pkl,
        "downstream_attack_pkl": paths.downstream_attack_pkl,
    }
    checkpoint_requirements = {
        "gen_original_tof_checkpoint": paths.gen_original_tof_checkpoint,
        "downstream_checkpoint": paths.downstream_checkpoint,
    }
    missing_data = tuple(missing_paths(data_requirements))
    missing_attack = tuple(missing_paths(attack_requirements))
    missing_checkpoint = tuple(missing_paths(checkpoint_requirements))
    if missing_data:
        status = STATUS_MISSING_DATA
    elif missing_attack:
        status = STATUS_MISSING_ATTACK
    elif missing_checkpoint:
        status = STATUS_MISSING_CHECKPOINT
    else:
        status = STATUS_DATA_READY
    return MatrixCellStatus(
        dataset=dataset,
        scenario=scenario,
        status=status,
        missing_data=missing_data,
        missing_attack=missing_attack,
        missing_checkpoint=missing_checkpoint,
        missing_generation=(),
        missing_downstream=(),
        paths=paths,
    )


def missing_paths(paths: dict[str, Path]) -> list[str]:
    return [f"{name}: {path}" for name, path in paths.items() if not path.exists()]


def load_reference_baseline(path: Path | None = None) -> dict[tuple[str, str], dict[str, Any]]:
    ref_path = path or Path(__file__).resolve().parent / "resources" / "reference" / "smartgen_table3_ad.json"
    payload = json.loads(ref_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", payload)
    return {(str(row["dataset"]), str(row["scenario"])): row for row in rows}
