# Recovery notes: SP-ST good locked version

This is the safest rollback candidate. It is based on the bundled SP-ST GPT-5.5 three-seed project copy and keeps the successful SP-ST result as the primary experiment.

Use this version when you need to preserve the result:

- proposed_causal_gss_gpt55_causal_tof mean F1: `0.975220`
- ablation_no_causal_tof mean F1: `0.898191`
- proposed mean FPR: `0.048535`

What I changed:

1. Added pytest isolation in `pyproject.toml` so frozen snapshots under `outputs/` are not collected as duplicate tests.
2. Added `scripts/check_recovery_integrity.py` to assert the bundled SP-ST metrics are still the expected good values.
3. Added `scripts/run_spst_locked_pipeline.sh` for a safe smoke path that does not expand to FR/SP/US x ST/TT/NT.
4. Added this note to make the rollback boundary explicit.

What I did not change:

- I did not regenerate GPT-5.5 JSONL.
- I did not rerun neural AD training in this environment.
- I did not change the stored metric JSON/CSV files.


## Evidence and positioning

This recovered copy was produced after comparing the two uploaded archives and the public GitHub branch pages. The key stability rule is: do not silently convert the successful SP-ST result into an all-dataset matrix run unless every dataset/scenario/seed has a validated GPT generation file and the Gen downstream data/checkpoints needed by that cell.

The included pytest configuration intentionally ignores `outputs/` and frozen code snapshots. Otherwise pytest may collect duplicate test modules from the frozen archive and fail during import collection even when the real tests pass.


## Recommended immediate command

```bash
cd CausalSmartHome_spst_good_locked_fixed
PYTHONPATH=. pytest -q
PYTHONPATH=. python scripts/check_recovery_integrity.py
```

Only run the frozen reproducer when you intentionally want to retrain downstream AD:

```bash
bash outputs/main_experiment_frozen/sp_st_gpt55_proposed_3seed_20260623/run_reproduce_from_frozen.sh
```
