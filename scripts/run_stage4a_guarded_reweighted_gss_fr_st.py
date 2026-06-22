#!/usr/bin/env python
from __future__ import annotations

import argparse
from stage4_common import add_stage4a_args, run_stage4a

SCENARIO = "fr_st"


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Run Stage 4A guarded causal-reweighted GSS generation-quality setup for {SCENARIO}.")
    add_stage4a_args(parser, SCENARIO)
    args = parser.parse_args()
    run_stage4a(args, SCENARIO)


if __name__ == "__main__":
    main()
