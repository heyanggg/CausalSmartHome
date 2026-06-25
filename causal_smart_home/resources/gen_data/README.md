# Gen Data Resources

This directory contains the SP-ST Gen data needed by the current main experiment:

- `dictionary.py`: device/action dictionary copied from the Gen source project.
- `sp/winter/trn.pkl`: SP winter source-context normal behavior data.
- `sp/spring/split_test.pkl`: SP spring target-context split test data.
- `sp/spring/test.pkl`, `trn.pkl`, `vld.pkl`: SP spring target-context data used by Gen evaluation code.

The runnable Gen code used by TOF and downstream AD is vendored under
`causal_smart_home/gen_core/`.

