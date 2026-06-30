from __future__ import annotations

import json
import math

import numpy as np

from causal_smart_home.json_utils import jsonable, write_json


def test_jsonable_converts_numpy_nan_to_none(tmp_path):
    payload = {"value": np.float64(math.nan)}

    converted = jsonable(payload)
    out = tmp_path / "payload.json"
    write_json(out, payload)

    assert converted == {"value": None}
    assert json.loads(out.read_text(encoding="utf-8")) == {"value": None}
