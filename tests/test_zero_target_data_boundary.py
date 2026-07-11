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


def test_main_prepare_generation_rejects_target_pkl() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/main_prepare_generation.py",
            "--dataset", "fr",
            "--scenario", "tt",
            "--seed", "2024",
            "--target-pkl", "target/test.pkl",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "unrecognized arguments: --target-pkl" in result.stderr


def test_prepare_commands_are_source_only(tmp_path: Path) -> None:
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
    assert not any(marker in rendered for marker in TARGET_MARKERS)


def test_generation_package_is_explicitly_target_free(tmp_path: Path, monkeypatch) -> None:
    module = load_script("build_codex_generation_package.py")
    source = tmp_path / "causal_gss"
    source.mkdir()
    (source / "prompt.txt").write_text("source-only prompt", encoding="utf-8")
    for name in ("causal_reweighted_gss_hints.json", "resolved_causal_relation_prior.json"):
        (source / name).write_text('{"target_data_used": false}', encoding="utf-8")
    out = tmp_path / "package"
    monkeypatch.setattr(sys, "argv", ["x", "--causal-gss-dir", str(source), "--out-dir", str(out), "--scenario", "fr_tt", "--seed", "2024"])
    module.main()

    schema = json.loads((out / "generation_schema.json").read_text(encoding="utf-8"))
    assert schema["target_data_used"] is False
    package_text = "\n".join(path.read_text(encoding="utf-8") for path in out.iterdir()).lower()
    assert not any(marker in package_text for marker in TARGET_MARKERS)


def test_generation_report_contract_is_source_only() -> None:
    source = (ROOT / "scripts" / "validate_and_pack_codex_generation.py").read_text(encoding="utf-8")
    assert '"target_data_used": False' in source
    assert '"target_pkl"' not in source
