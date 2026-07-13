from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET_MARKERS = ("test.pkl", "split_test.pkl", "target_test", "target-pkl")


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_prepare_generation_accepts_target_pkl() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/main_prepare_generation.py",
            "--dataset", "fr",
            "--scenario", "tt",
            "--seed", "2024",
            "--target-pkl", "target/test.pkl",
            "--dry-run-command",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "--target-pkl target/test.pkl" in result.stdout


def test_prepare_commands_include_target_aware_outputs(tmp_path: Path) -> None:
    module = load_script("main_prepare_generation.py")
    args = module.parse_args_from([
        "--dataset", "fr",
        "--scenario", "tt",
        "--seed", "2024",
        "--out-root", str(tmp_path),
    ]) if hasattr(module, "parse_args_from") else None
    if args is None:
        old = sys.argv
        try:
            sys.argv = [old[0], "--dataset", "fr", "--scenario", "tt", "--seed", "2024", "--out-root", str(tmp_path)]
            args = module.parse_args()
        finally:
            sys.argv = old
    rendered = " ".join(part for command in module.build_commands(args) for part in command).lower()
    assert "daytime/trn.pkl" in rendered
    assert "--target-pkl" in rendered
    assert "target_adapted_causal_prior.json" in rendered


def test_generation_package_records_target_aware_artifacts(tmp_path: Path, monkeypatch) -> None:
    module = load_script("build_codex_generation_package.py")
    source = tmp_path / "causal_gss"
    source.mkdir()
    (source / "prompt.txt").write_text("source-only prompt", encoding="utf-8")
    for name in ("causal_reweighted_gss_hints.json", "resolved_causal_relation_prior.json"):
        (source / name).write_text('{"target_data_used": false}', encoding="utf-8")
    for name in ("target_adapted_causal_prior.json", "guard_report.json"):
        (source / name).write_text("{}", encoding="utf-8")
    out = tmp_path / "package"
    monkeypatch.setattr(sys, "argv", ["x", "--causal-gss-dir", str(source), "--out-dir", str(out), "--scenario", "fr_tt", "--seed", "2024"])
    module.main()

    schema = json.loads((out / "generation_schema.json").read_text(encoding="utf-8"))
    assert schema["target_data_used"] is True
    assert (out / "target_adapted_causal_prior.json").exists()


def test_generation_report_contract_keeps_target_usage_metadata() -> None:
    source = (ROOT / "scripts" / "validate_and_pack_codex_generation.py").read_text(encoding="utf-8")
    assert '"target_data_used": False' in source
    assert '"target_pkl"' not in source
