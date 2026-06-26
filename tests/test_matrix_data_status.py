import pickle

from causal_smart_home.experiment_matrix import (
    STATUS_DATA_READY,
    STATUS_MISSING_DATA,
    check_matrix_cell_data_ready,
    resolve_existing_context,
)


def _dump(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump([[1, 2, 3, 4]], f)


def _make_ready_cell(root, dataset="fr"):
    base = root / "causal_smart_home" / "resources" / "gen_data"
    _dump(base / "dictionary.py")
    for path in [
        base / dataset / "winter" / "trn.pkl",
        base / dataset / "spring" / "trn.pkl",
        base / dataset / "spring" / "vld.pkl",
        base / dataset / "spring" / "test.pkl",
        base / dataset / "spring" / "split_test.pkl",
        root / "causal_smart_home" / "gen_core" / "anomaly_detection_pipeline" / "test" / dataset / "spring" / "test.pkl",
        root / "causal_smart_home" / "gen_core" / "anomaly_detection_pipeline" / "attack" / dataset / f"labeled_{dataset}_spring_attack_heater.pkl",
        root / "causal_smart_home" / "gen_core" / "gen_original_tof" / "check_model" / f"best_{dataset}_gpt-4o_SPPC.pth",
        root / "causal_smart_home" / "gen_core" / "anomaly_detection_pipeline" / "check_model" / f"best_{dataset}_gpt-4o_SPPC.pth",
    ]:
        _dump(path)


def test_check_matrix_cell_data_ready_distinguishes_ready_and_missing(tmp_path):
    _make_ready_cell(tmp_path)

    ready = check_matrix_cell_data_ready("fr", "st", tmp_path)
    missing = check_matrix_cell_data_ready("sp", "st", tmp_path)

    assert ready.status == STATUS_DATA_READY
    assert not ready.missing
    assert missing.status == STATUS_MISSING_DATA
    assert missing.missing_data


def test_context_aliases_resolve_day_night_multi(tmp_path):
    root = tmp_path / "gen_data"
    (root / "fr" / "day").mkdir(parents=True)
    (root / "fr" / "night").mkdir(parents=True)
    (root / "fr" / "multi").mkdir(parents=True)

    assert resolve_existing_context(root, "fr", "daytime") == "day"
    assert resolve_existing_context(root, "fr", "nighttime") == "night"
    assert resolve_existing_context(root, "fr", "multiple") == "multi"


def test_nt_multiple_does_not_require_target_train_validation(tmp_path):
    base = tmp_path / "causal_smart_home" / "resources" / "gen_data"
    _dump(base / "dictionary.py")
    for path in [
        base / "us" / "single" / "trn.pkl",
        base / "us" / "multiple" / "test.pkl",
        base / "us" / "multiple" / "split_test.pkl",
        tmp_path / "causal_smart_home" / "gen_core" / "anomaly_detection_pipeline" / "test" / "us" / "multiple" / "test.pkl",
        tmp_path / "causal_smart_home" / "gen_core" / "anomaly_detection_pipeline" / "attack" / "us" / "labeled_us_multiple_attack_tv.pkl",
        tmp_path / "causal_smart_home" / "gen_core" / "gen_original_tof" / "check_model" / "best_us_gpt-4o_SPPC.pth",
        tmp_path / "causal_smart_home" / "gen_core" / "anomaly_detection_pipeline" / "check_model" / "best_us_gpt-4o_SPPC.pth",
    ]:
        _dump(path)

    status = check_matrix_cell_data_ready("us", "nt", tmp_path)

    assert status.status == STATUS_DATA_READY
    assert not any("target_train_pkl" in item for item in status.missing)
    assert not any("target_validation_pkl" in item for item in status.missing)
