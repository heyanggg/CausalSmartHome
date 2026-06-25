from causal_smart_home.experiment_matrix import DATASETS, SCENARIOS, SEEDS, experiment_grid


def test_experiment_matrix_has_9_dataset_scenario_cells():
    grid = experiment_grid()

    assert len(grid) == 9
    assert {(item.dataset, item.scenario) for item in grid} == {
        (dataset, scenario) for dataset in DATASETS for scenario in SCENARIOS
    }
    assert SEEDS == [2024, 2025, 2026]


def test_experiment_matrix_contexts_are_canonical():
    for item in experiment_grid():
        assert item.dataset in DATASETS
        assert item.scenario in SCENARIOS
        assert item.source_context
        assert item.target_context
        assert item.source_aliases
        assert item.target_aliases
