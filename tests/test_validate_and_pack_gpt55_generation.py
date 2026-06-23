from __future__ import annotations

import json
import pickle

import pytest

from scripts.validate_and_pack_gpt55_generation import main


def write_dictionary(path):
    path.write_text(
        "\n".join(
            [
                "sp_devices_dict = {'Fan': 11, 'Light': 13}",
                "sp_actions = {'Fan:switch on': 68, 'Light:switch on': 80}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_validate_and_pack_gpt55_generation_valid(tmp_path, monkeypatch):
    dictionary_py = tmp_path / "dictionary.py"
    write_dictionary(dictionary_py)
    jsonl = tmp_path / "generated.jsonl"
    jsonl.write_text(
        json.dumps({"sequence_id": "seq_000", "sequence": [1, 0, 11, 68, 1, 1, 13, 80], "notes": "fan then light"}) + "\n",
        encoding="utf-8",
    )
    out_pkl = tmp_path / "generated.pkl"
    validation_report = tmp_path / "validation_report.json"
    generation_report = tmp_path / "generation_report.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_and_pack_gpt55_generation.py",
            "--input-jsonl",
            str(jsonl),
            "--out-pkl",
            str(out_pkl),
            "--out-validation-report",
            str(validation_report),
            "--out-generation-report",
            str(generation_report),
            "--dictionary-py",
            str(dictionary_py),
            "--dataset",
            "sp",
            "--scenario",
            "st",
            "--scenario-key",
            "sp_st",
            "--seed",
            "2024",
            "--expected-count",
            "1",
            "--expected-length",
            "8",
        ],
    )

    main()

    assert pickle.loads(out_pkl.read_bytes()) == [[1, 0, 11, 68, 1, 1, 13, 80]]
    assert json.loads(validation_report.read_text(encoding="utf-8"))["status"] == "valid"
    report = json.loads(generation_report.read_text(encoding="utf-8"))
    assert report["generator"] == "gpt55_generation"
    assert report["generation_model"] == "GPT-5.5"
    assert report["api_llm"] is False
    assert report["manual_generation"] is True
    assert report["surrogate_algorithm"] is False


def test_validate_and_pack_gpt55_generation_rejects_mismatch(tmp_path, monkeypatch):
    dictionary_py = tmp_path / "dictionary.py"
    write_dictionary(dictionary_py)
    jsonl = tmp_path / "generated.jsonl"
    jsonl.write_text(json.dumps({"sequence_id": "bad", "sequence": [1, 0, 11, 80]}) + "\n", encoding="utf-8")
    validation_report = tmp_path / "validation_report.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_and_pack_gpt55_generation.py",
            "--input-jsonl",
            str(jsonl),
            "--out-pkl",
            str(tmp_path / "generated.pkl"),
            "--out-validation-report",
            str(validation_report),
            "--out-generation-report",
            str(tmp_path / "generation_report.json"),
            "--dictionary-py",
            str(dictionary_py),
            "--dataset",
            "sp",
            "--scenario",
            "st",
            "--scenario-key",
            "sp_st",
            "--seed",
            "2024",
            "--expected-count",
            "1",
            "--expected-length",
            "4",
        ],
    )

    with pytest.raises(ValueError, match="validation failed"):
        main()
    report = json.loads(validation_report.read_text(encoding="utf-8"))
    assert report["status"] == "invalid"
    assert "device_action_mismatch" in report["invalid_sequences"][0]["reasons"][0]
