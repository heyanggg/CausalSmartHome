import importlib.util
from pathlib import Path

import pytest

from causal_smart_home.cli import main


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is required for the demo prior training")


def test_cli_demo(tmp_path):
    out = tmp_path / "demo"
    main(["demo", "--out-dir", str(out), "--num-sequences", "10", "--epochs", "1", "--lag", "2"])
    assert (out / "causal_prior.json").exists()
    assert (out / "causal_smartgen_prompt.txt").exists()
    assert (out / "causal_filter_scores.json").exists()
