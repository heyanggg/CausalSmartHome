import json
import pickle
import textwrap

from causal_smart_home.smartgen_original_tof import SmartGenOriginalTOFConfig, run_smartgen_original_tof


def _dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def test_smartgen_original_tof_wrapper_calls_mock_security_check(tmp_path):
    smartgen_code = tmp_path / "SmartGen"
    smartgen_code.mkdir()
    (smartgen_code / "security_check.py").write_text(
        textwrap.dedent(
            """
            import pickle
            from pathlib import Path

            def setup_seed(seed):
                pass

            def security_check(dataset, new_env, thres, method, model):
                base = f"{dataset}_{new_env}_generation_{method}_th={thres}_{model}_seq"
                root = Path("filter_data") / dataset / new_env
                data = pickle.load(open(root / f"{base}.pkl", "rb"))
                # Mimic original two-stage TOF final output: retain all but last.
                with open(root / f"{base}_filter_true.pkl", "wb") as f:
                    pickle.dump(data[:-1], f)
            """
        ),
        encoding="utf-8",
    )
    generated = tmp_path / "fresh.pkl"
    _dump(generated, [[i, i, i, i] for i in range(4)])

    report = run_smartgen_original_tof(
        SmartGenOriginalTOFConfig(
            smartgen_root=tmp_path,
            security_check_path=smartgen_code / "security_check.py",
            dataset="fr",
            scenario="st",
            generated_pkl=generated,
            out_pkl=tmp_path / "out" / "smartgen_tof.pkl",
            out_dir=tmp_path / "out",
            seed=2024,
        )
    )

    assert report["status"] == "success"
    assert report["used_smartgen_original_tof"] is True
    assert report["num_generated_before_tof"] == 4
    assert report["num_generated_after_smartgen_tof"] == 3
    assert pickle.load(open(tmp_path / "out" / "smartgen_tof.pkl", "rb")) == [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]
    saved = json.loads((tmp_path / "out" / "smartgen_original_tof_report.json").read_text())
    assert saved["smartgen_original_tof_stage1"] == "reconstruction_loss_iqr_outlier_detection"
    assert saved["smartgen_original_tof_stage2"] == "utility_value_selection"
