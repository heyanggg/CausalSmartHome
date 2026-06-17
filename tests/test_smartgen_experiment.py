import json
import pickle

from causal_smart_home.smartgen_experiment import (
    SmartGenAnomalyRunConfig,
    default_smartgen_paths,
    default_synthetic_pkl,
    run_smartgen_anomaly_experiment,
    split_random_to_files,
)


def _dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def test_default_smartgen_paths_and_synthetic_name(tmp_path):
    paths = default_smartgen_paths(tmp_path, "fr", "spring")
    synthetic = default_synthetic_pkl(tmp_path, "fr", "spring")

    assert paths["attack_pkl"].as_posix().endswith("attack/fr/labeled_fr_spring_attack_heater.pkl")
    assert paths["target_test_pkl"].as_posix().endswith("test/fr/spring/test.pkl")
    assert synthetic.name == "fr_spring_generation_SPPC_th=0.918_gpt-4o_seq_filter_true.pkl"


def test_split_random_to_files_is_deterministic(tmp_path):
    data = [[i] for i in range(10)]
    src = tmp_path / "src.pkl"
    train = tmp_path / "train.pkl"
    vld = tmp_path / "vld.pkl"
    _dump(src, data)

    first_train, first_vld = split_random_to_files(src, train, vld, split_ratio=0.8, seed=7)
    second_train, second_vld = split_random_to_files(src, train, vld, split_ratio=0.8, seed=7)

    assert len(first_train) == 8
    assert len(first_vld) == 2
    assert first_train == second_train
    assert first_vld == second_vld


def test_smartgen_anomaly_dry_run_writes_payload_and_split(tmp_path):
    synthetic = tmp_path / "synthetic.pkl"
    _dump(synthetic, [[i] * 4 for i in range(5)])
    attack = tmp_path / "attack.pkl"
    target = tmp_path / "target.pkl"
    _dump(attack, [([1, 2, 3, 4], 1)])
    _dump(target, [[1, 2, 3, 4]])
    validation = tmp_path / "validation.pkl"
    _dump(validation, [[2, 3, 4, 5], [3, 4, 5, 6]])

    config = SmartGenAnomalyRunConfig(
        smartgen_root=tmp_path,
        dataset="fr",
        env="spring",
        synthetic_pkl=synthetic,
        out_dir=tmp_path / "out",
        tag="dry",
        cuda_visible_devices="0",
        dry_run=True,
        attack_pkl=attack,
        target_test_pkl=target,
        validation_pkl=validation,
    )

    payload = run_smartgen_anomaly_experiment(config)

    assert payload["synthetic_size"] == 5
    assert payload["train_size"] == 4
    assert payload["vld_size"] == 1
    assert payload["threshold_vld_size"] == 2
    assert payload["requested_device"] == "cuda"
    assert payload["cuda_visible_devices"] == "0"
    assert (tmp_path / "out" / "dry_train.pkl").exists()
    assert (tmp_path / "out" / "dry_vld.pkl").exists()
    saved = json.loads((tmp_path / "out" / "dry_smartgen_anomaly_eval.json").read_text())
    assert saved["dry_run"] is True
    assert saved["threshold_vld_pkl"] == str(validation.resolve())
