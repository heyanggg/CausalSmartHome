# Gen Data Resources

This directory contains the vendored Gen data needed by the CausalSmartHome main
experiment matrix:

```text
FR/SP/US x spring/night/multiple
```

- `dictionary.py`: device/action dictionary copied from the Gen source project.
- `{fr,sp,us}/winter/`: source-context data for spring transfer.
- `{fr,sp,us}/daytime/`: source-context data for night transfer.
- `{fr,sp,us}/single/`: source-context data for multiple transfer.
- `{fr,sp,us}/{spring,night,multiple}/`: target-context test/split data and any
  target auxiliary files provided by Gen.

The runnable Gen code used by TOF and downstream AD is vendored under
`causal_smart_home/gen_core/`.

Use `python scripts/check_gen_main_data.py` from the project root to verify the
required Gen data, checkpoints, attacks, tests, and reference result files.
