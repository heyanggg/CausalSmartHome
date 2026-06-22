#!/usr/bin/env python
from __future__ import annotations

import argparse
from stage4_common import add_stage4b_args, run_stage4b

SCENARIO = "sp_st"
WEIGHTED = bool(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Register or dry-run Stage 4B downstream AD sanity-check metrics for {SCENARIO}.")
    add_stage4b_args(parser, SCENARIO, weighted=WEIGHTED)
    args = parser.parse_args()
    run_stage4b(args, SCENARIO, weighted=WEIGHTED)


if __name__ == "__main__":
    main()
