# CASE-SHU-OSHER-DG evidence chain

| item | path |
|---|---|
| impl (case definition, library) | `tensordg/cases/shock_entropy.py` |
| impl (DG ladder / merge driver) | `benchmarks/shu_osher_case_pool.py` |
| impl (independent reference solver) | `benchmarks/fv_weno5_1d.py` |
| entry | `python -m benchmarks.shu_osher_case_pool merge` |
| repro | `bash validation/repro_shu_osher/run_all.sh` |
| run artifacts | `validation/repro_shu_osher/artifacts/{metrics.json,profiles.npz,raw/*.npz}` |
| benchmark ref vs got | `validation/repro_shu_osher/benchmark_ref_vs_got.json` |
| report | `docs/reports/shu-osher-dg-case-pool.md` |
| tests | `tests/test_shu_osher_case_pool.py`, `tests/test_shock_entropy_case.py` |
| integration caller | `benchmarks/shu_osher_1d.py`, `benchmarks/shu_osher_case_pool.py` import `tensordg.cases.shock_entropy` |

All 9 DG configurations: PASS the 5% acceptance line against the independent reference.

Artifacts and sha256: see `artifacts.sha256.json`.

Honest gap: the paper's own plotted values are WITHHELD (no PDF reachable); the quantitative reference is the independent FV-WENO5 solution plus the exact Rankine-Hugoniot anchors.
