from __future__ import annotations

import json
from pathlib import Path

import scripts.check_project as check_project


def test_project_doctor_checks_proposed_coordinates(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(check_project, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(check_project, "REQUIRED_PROJECT_FILES", ())
    monkeypatch.setattr(check_project, "DATASETS", ("fr",))
    monkeypatch.setattr(check_project, "ENVIRONMENTS", ("spring",))
    monkeypatch.setattr(check_project, "SCENARIO_BY_ENV", {"spring": "st"})
    monkeypatch.setattr(check_project, "build_gen_data_report", lambda: {"status": "ok", "missing": []})
    metrics = (
        tmp_path
        / "fr_st"
        / "seed2024"
        / "downstream_ad"
        / check_project.PROPOSED_VARIANT
        / "normalized_metrics.json"
    )
    metrics.parent.mkdir(parents=True)
    metrics.write_text(
        json.dumps(
            {
                "status": "success",
                "dataset": "fr",
                "scenario": "st",
                "seed": 2024,
                "variant": check_project.PROPOSED_VARIANT,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "fpr": 0.0,
                "device": "cuda",
                "requested_device": "cuda",
            }
        ),
        encoding="utf-8",
    )

    report = check_project.build_report(tmp_path)

    assert report["status"] == "ok"
    assert report["checked_proposed_metrics"] == 1


def test_project_doctor_reports_coordinate_mismatch(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(check_project, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(check_project, "REQUIRED_PROJECT_FILES", ())
    monkeypatch.setattr(check_project, "DATASETS", ("fr",))
    monkeypatch.setattr(check_project, "ENVIRONMENTS", ("spring",))
    monkeypatch.setattr(check_project, "SCENARIO_BY_ENV", {"spring": "st"})
    monkeypatch.setattr(check_project, "build_gen_data_report", lambda: {"status": "ok", "missing": []})
    metrics = (
        tmp_path
        / "fr_st"
        / "seed2024"
        / "downstream_ad"
        / check_project.PROPOSED_VARIANT
        / "normalized_metrics.json"
    )
    metrics.parent.mkdir(parents=True)
    metrics.write_text(json.dumps({"dataset": "sp", "seed": 9, "variant": "wrong"}), encoding="utf-8")

    report = check_project.build_report(tmp_path)

    kinds = {issue["kind"] for issue in report["issues"]}
    assert "missing_metric_fields" in kinds
    assert "variant_mismatch" in kinds
    assert "experiment_coordinate_mismatch" in kinds
