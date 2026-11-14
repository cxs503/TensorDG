# PENDING: reference re-run

`metrics.json` / `summary.md` / `error_conservation.csv` in this directory are the
*pre-fix* evidence bundle: every error there is measured against the free-space
exact solution, which is not a solution of the periodic problem for t >~ 4
(see `docs/CASE-DG-EULER-VORTEX-ERR-reference-defect.md`).

Do not cite them as the post-fix result. The 12-checkpoint ladder must be re-solved
with the periodic reference (`PERIOD` in `benchmarks/case_euler_vortex_errconserve.py`);
submitted as run_compute job cj2908, still queued at hand-off.
Regenerate with:

    PYTHONPATH=. python3 benchmarks/case_euler_vortex_errconserve.py --aggregate --figures \
        --output-dir case_euler_vortex_errconserve
