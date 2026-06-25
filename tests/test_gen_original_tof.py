import json
import pickle
import textwrap

from causal_smart_home.gen_original_tof import GenOriginalTOFConfig, run_gen_original_tof


def _dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def test_gen_original_tof_wrapper_calls_mock_security_check(tmp_path):
    gen_code = tmp_path / "Gen"
    gen_code.mkdir()
    (gen_code / "security_check.py").write_text(
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
                # Mimic original two-pass TOF final output: retain all but last.
                with open(root / f"{base}_filter_true.pkl", "wb") as f:
                    pickle.dump(data[:-1], f)
            """
        ),
        encoding="utf-8",
    )
    generated = tmp_path / "fresh.pkl"
    _dump(generated, [[i, i, i, i] for i in range(4)])

    report = run_gen_original_tof(
        GenOriginalTOFConfig(
            gen_root=tmp_path,
            security_check_path=gen_code / "security_check.py",
            dataset="sp",
            scenario="st",
            generated_pkl=generated,
            out_pkl=tmp_path / "out" / "gen_tof.pkl",
            out_dir=tmp_path / "out",
            seed=2024,
        )
    )

    assert report["status"] == "success"
    assert report["used_gen_original_tof"] is True
    assert report["num_generated_before_tof"] == 4
    assert report["num_generated_after_gen_tof"] == 3
    assert pickle.load(open(tmp_path / "out" / "gen_tof.pkl", "rb")) == [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]
    saved = json.loads((tmp_path / "out" / "gen_original_tof_report.json").read_text())
    assert saved["gen_original_tof_filter"] == "reconstruction_loss_iqr_outlier_detection"
    assert saved["gen_original_tof_utility_selection"] == "utility_value_selection"
