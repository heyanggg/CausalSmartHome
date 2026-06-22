import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from causal_smart_home.schema import BehaviorEvent, BehaviorSequence

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "verify_and_repair_generated_sequences.py"
SPEC = importlib.util.spec_from_file_location("verify_and_repair_generated_sequences", SCRIPT_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)

BUILD_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_guarded_causal_reweighted_gss_prompt.py"
BUILD_SPEC = importlib.util.spec_from_file_location("build_guarded_causal_reweighted_gss_prompt", BUILD_SCRIPT_PATH)
build_module = importlib.util.module_from_spec(BUILD_SPEC)
assert BUILD_SPEC.loader is not None
BUILD_SPEC.loader.exec_module(build_module)


def test_repair_prompt_contains_violated_edge_and_overused_device():
    seq = BehaviorSequence([BehaviorEvent(0, 0, 2, 20), BehaviorEvent(0, 1, 1, 10)])
    score = {"violated_edges": [{"source_name": "Light", "target_name": "Television"}]}
    prompt = module.build_repair_prompt(seq, score, overused_devices=[{"device_name": "Television"}], target_distribution_warning={"d:2": 0.1})
    assert "violated causal edges" in prompt.lower()
    assert "Television" in prompt
    assert "minimal changes" in prompt
    assert "same format" in prompt


def test_new_stage4_scripts_have_help():
    scripts = [
        "build_guarded_causal_reweighted_gss_prompt.py",
        "run_causal_tof_weighting.py",
        "verify_and_repair_generated_sequences.py",
        "run_stage4a_guarded_reweighted_gss_fr_st.py",
        "run_stage4b_ad_guarded_reweighted_gss_sp_st.py",
        "summarize_stage4_guarded_reweighted.py",
        "summarize_stage4_causal_tof.py",
        "build_codex_gpt55_generation_package.py",
        "generate_codex_gpt55_surrogate_sequences.py",
        "validate_generated_sequences.py",
        "summarize_stage4_downweight_codex_gpt55.py",
        "summarize_stage4_downstream_ad.py",
    ]
    root = Path(__file__).resolve().parents[1]
    for script in scripts:
        result = subprocess.run([sys.executable, str(root / "scripts" / script), "--help"], cwd=root, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()


def test_stage4b_help_accepts_provenance_args():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "run_stage4b_ad_causal_tof_weighted_sp_st.py"), "--help"],
        cwd=root,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--stage4a-dir" in result.stdout
    assert "--weighted-generated-pkl" in result.stdout


def test_guard_report_overused_devices_get_readable_names():
    report = {"overused_devices": [{"device_key": "d:30", "device_name": "device_30"}]}
    annotated = build_module.annotate_guard_report_device_names(report, {30: "Television"})
    assert annotated["overused_devices"][0]["device_name"] == "Television"
