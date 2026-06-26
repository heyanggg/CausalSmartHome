import json
import pickle

from scripts.migrate_gen_experiment_data import main as migrate_main


def _pkl(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump([[1, 2, 3, 4]], f)


def _text(path, text="x = 1\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_fake_smartgen(root):
    source = root / "SmartGen"
    for path in [
        source / "SmartGen" / "IoT_data" / "fr" / "winter" / "trn.pkl",
        source / "SmartGen" / "IoT_data" / "fr" / "spring" / "trn.pkl",
        source / "SmartGen" / "IoT_data" / "fr" / "spring" / "vld.pkl",
        source / "SmartGen" / "IoT_data" / "fr" / "spring" / "test.pkl",
        source / "SmartGen" / "IoT_data" / "fr" / "spring" / "split_test.pkl",
        source / "anomaly_detection_pipeline" / "test" / "fr" / "spring" / "test.pkl",
        source / "anomaly_detection_pipeline" / "attack" / "fr" / "labeled_fr_spring_attack_heater.pkl",
        source / "anomaly_detection_pipeline" / "attack" / "fr" / "fr_spring_attack_heater.pkl",
        source / "SmartGen" / "check_model" / "best_fr_gpt-4o_SPPC.pth",
        source / "anomaly_detection_pipeline" / "check_model" / "best_fr_gpt-4o_SPPC.pth",
    ]:
        _pkl(path)
    _text(source / "SmartGen" / "dictionary.py")
    _text(source / "SmartGen" / "security_check.py")
    _text(source / "anomaly_detection_pipeline" / "models1.py")
    return source


def _run(monkeypatch, source, target, *extra, dataset="fr", scenario="st"):
    monkeypatch.setattr(
        "sys.argv",
        [
            "migrate_gen_experiment_data.py",
            "--source",
            str(source),
            "--target",
            str(target),
            "--matrix",
            "single",
            "--dataset",
            dataset,
            "--scenario",
            scenario,
            "--write-manifest",
            *extra,
        ],
    )
    migrate_main()


def test_migration_manifest_dry_run_and_copy(tmp_path, monkeypatch):
    source = _make_fake_smartgen(tmp_path)
    target = tmp_path / "CausalSmartHome"
    target.mkdir()

    _run(monkeypatch, source, target, "--dry-run")
    dry_manifest = json.loads((target / "outputs" / "data_migration" / "gen_data_migration_manifest.json").read_text())
    assert dry_manifest["summary"]["status_counts"]["SKIPPED"] > 0
    assert "COPIED" not in dry_manifest["summary"]["status_counts"]

    _run(monkeypatch, source, target, "--copy")
    copy_manifest = json.loads((target / "outputs" / "data_migration" / "gen_data_migration_manifest.json").read_text())
    assert copy_manifest["summary"]["status_counts"]["COPIED"] > 0
    assert (target / "causal_smart_home" / "resources" / "gen_data" / "fr" / "spring" / "split_test.pkl").exists()

    _run(monkeypatch, source, target, "--copy")
    exists_manifest = json.loads((target / "outputs" / "data_migration" / "gen_data_migration_manifest.json").read_text())
    assert exists_manifest["summary"]["status_counts"]["EXISTS"] > 0


def test_migration_accepts_train_and_validation_aliases(tmp_path, monkeypatch):
    source = tmp_path / "SmartGen"
    for path in [
        source / "SmartGen" / "IoT_data" / "us" / "winter" / "train.pkl",
        source / "SmartGen" / "IoT_data" / "us" / "spring" / "train.pkl",
        source / "SmartGen" / "IoT_data" / "us" / "spring" / "rs_vld.pkl",
        source / "SmartGen" / "IoT_data" / "us" / "spring" / "test.pkl",
        source / "SmartGen" / "IoT_data" / "us" / "spring" / "split_test.pkl",
        source / "anomaly_detection_pipeline" / "test" / "us" / "spring" / "test.pkl",
        source / "anomaly_detection_pipeline" / "attack" / "us" / "labeled_us_spring_attack_heater.pkl",
        source / "anomaly_detection_pipeline" / "attack" / "us" / "us_spring_attack_heater.pkl",
        source / "SmartGen" / "check_model" / "best_us_gpt-4o_SPPC.pth",
        source / "anomaly_detection_pipeline" / "check_model" / "best_us_gpt-4o_SPPC.pth",
    ]:
        _pkl(path)
    _text(source / "SmartGen" / "dictionary.py")
    _text(source / "SmartGen" / "security_check.py")
    _text(source / "anomaly_detection_pipeline" / "models1.py")

    target = tmp_path / "CausalSmartHome"
    target.mkdir()

    _run(monkeypatch, source, target, "--copy", dataset="us", scenario="st")
    manifest_path = target / "outputs" / "data_migration" / "gen_data_migration_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    assert (target / "causal_smart_home" / "resources" / "gen_data" / "us" / "winter" / "trn.pkl").exists()
    assert (target / "causal_smart_home" / "resources" / "gen_data" / "us" / "spring" / "trn.pkl").exists()
    assert (target / "causal_smart_home" / "resources" / "gen_data" / "us" / "spring" / "vld.pkl").exists()
    alias_rows = [row for row in manifest["entries"] if row["alias_used"]]
    assert {row["target_path"].split("/")[-1] for row in alias_rows} >= {"trn.pkl", "vld.pkl"}
    assert any(row["source_path"].endswith("rs_vld.pkl") for row in alias_rows)


def test_migration_marks_true_source_missing(tmp_path, monkeypatch):
    source = tmp_path / "SmartGen"
    target = tmp_path / "CausalSmartHome"
    source.mkdir()
    target.mkdir()

    _run(monkeypatch, source, target, "--dry-run")
    manifest = json.loads((target / "outputs" / "data_migration" / "gen_data_migration_manifest.json").read_text())

    assert manifest["summary"]["status_counts"]["SOURCE_MISSING"] > 0
    assert "MISSING" not in manifest["summary"]["status_counts"]
