import subprocess
import sys


def help_text(script):
    result = subprocess.run([sys.executable, script, "--help"], text=True, capture_output=True, check=True)
    return result.stdout


def test_main_scripts_expose_dataset_scenario_and_matrix_args():
    matrix_help = help_text("scripts/run_main_experiment_matrix.py")
    summary_help = help_text("scripts/summarize_main_experiment.py")
    gen_tof_help = help_text("scripts/run_gen_original_tof.py")
    downstream_help = help_text("scripts/run_gen_downstream_ad.py")

    assert "--matrix" in matrix_help
    assert "--dataset" in gen_tof_help
    assert "--scenario" in gen_tof_help
    assert "--dataset" in downstream_help
    assert "--scenario" in downstream_help
    assert "--matrix" in summary_help


def test_summary_help_does_not_describe_sp_st_only_scan():
    summary_help = help_text("scripts/summarize_main_experiment.py")

    assert "SP-ST" not in summary_help
    assert "sp_st" not in summary_help


def test_matrix_dry_run_scans_all_cells():
    result = subprocess.run(
        [sys.executable, "scripts/run_main_experiment_matrix.py", "--dry-run", "--matrix", "all"],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "matrix cells: 27" in result.stdout
