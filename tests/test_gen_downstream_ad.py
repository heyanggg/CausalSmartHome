import json
import pickle

from causal_smart_home.gen_downstream_ad import (
    DEFAULT_THRESHOLD_PERCENTAGES,
    DEFAULT_THRESHOLDS,
    GenDownstreamADRunConfig,
    default_gen_paths,
    env_for_scenario,
    run_gen_downstream_ad_experiment,
    split_generated_to_train_validation,
)


def _dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def test_default_gen_paths(tmp_path):
    paths = default_gen_paths(tmp_path, "sp", "spring")

    assert paths["attack_pkl"].as_posix().endswith("attack/sp/labeled_sp_spring_attack_heater.pkl")
    assert paths["target_test_pkl"].as_posix().endswith("test/sp/spring/test.pkl")

    fr_paths = default_gen_paths(tmp_path, "fr", "night")
    assert fr_paths["attack_pkl"].as_posix().endswith("attack/fr/labeled_fr_night_attack_time.pkl")
    assert fr_paths["target_test_pkl"].as_posix().endswith("test/fr/night/test.pkl")


def test_gen_main_matrix_thresholds_and_scenario_aliases():
    assert len(DEFAULT_THRESHOLDS) == 9
    assert len(DEFAULT_THRESHOLD_PERCENTAGES) == 9
    assert env_for_scenario("st") == "spring"
    assert env_for_scenario("tt") == "night"
    assert env_for_scenario("nt") == "multiple"
    assert DEFAULT_THRESHOLDS[("us", "multiple")] == "0.913"


def test_split_generated_to_train_validation_is_deterministic(tmp_path):
    data = [[i] for i in range(10)]
    src = tmp_path / "src.pkl"
    train = tmp_path / "train.pkl"
    vld = tmp_path / "vld.pkl"
    _dump(src, data)

    first_train, first_vld = split_generated_to_train_validation(src, train, vld, split_ratio=0.8, seed=7)
    second_train, second_vld = split_generated_to_train_validation(src, train, vld, split_ratio=0.8, seed=7)

    assert len(first_train) == 8
    assert len(first_vld) == 2
    assert first_train == second_train
    assert first_vld == second_vld


def test_gen_anomaly_dry_run_writes_payload_and_split(tmp_path):
    synthetic = tmp_path / "synthetic.pkl"
    _dump(synthetic, [[i] * 4 for i in range(5)])
    attack = tmp_path / "attack.pkl"
    target = tmp_path / "target.pkl"
    _dump(attack, [([1, 2, 3, 4], 1)])
    _dump(target, [[1, 2, 3, 4]])
    validation = tmp_path / "validation.pkl"
    _dump(validation, [[2, 3, 4, 5], [3, 4, 5, 6]])

    config = GenDownstreamADRunConfig(
        gen_root=tmp_path,
        dataset="sp",
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

    payload = run_gen_downstream_ad_experiment(config)

    assert payload["synthetic_size"] == 5
    assert payload["train_size"] == 4
    assert payload["vld_size"] == 1
    assert payload["threshold_vld_size"] == 2
    assert payload["requested_device"] == "cuda"
    assert payload["cuda_visible_devices"] == "0"
    assert (tmp_path / "out" / "dry_train.pkl").exists()
    assert (tmp_path / "out" / "dry_vld.pkl").exists()
    saved = json.loads((tmp_path / "out" / "dry_gen_downstream_ad_eval.json").read_text())
    assert saved["dry_run"] is True
    assert saved["threshold_vld_pkl"] == str(validation.resolve())


def test_gen_multiple_dry_run_uses_full_synthetic_for_train_and_validation(tmp_path):
    synthetic = tmp_path / "synthetic.pkl"
    _dump(synthetic, [[i] * 4 for i in range(5)])
    attack = tmp_path / "attack.pkl"
    target = tmp_path / "target.pkl"
    _dump(attack, [([1, 2, 3, 4], 1)])
    _dump(target, [[1, 2, 3, 4]])

    config = GenDownstreamADRunConfig(
        gen_root=tmp_path,
        dataset="sp",
        env="multiple",
        synthetic_pkl=synthetic,
        out_dir=tmp_path / "out",
        tag="dry_multiple",
        cuda_visible_devices="0",
        dry_run=True,
        attack_pkl=attack,
        target_test_pkl=target,
    )

    payload = run_gen_downstream_ad_experiment(config)

    assert payload["training_protocol"] == "smartgen_multiple_full_synthetic_train_and_validation"
    assert payload["train_size"] == 5
    assert payload["vld_size"] == 5
    assert payload["threshold_vld_size"] == 5
    assert payload["train_pkl"] == str(synthetic.resolve())
    assert payload["threshold_vld_pkl"] == str(synthetic.resolve())
    assert not (tmp_path / "out" / "dry_multiple_train.pkl").exists()
    assert not (tmp_path / "out" / "dry_multiple_vld.pkl").exists()
