# Sharded C2 ladder -- execution-path proof (smoke)

`legs/` holds two real leg artifacts (`p1_32_t0.01_cfl0.2`, `p1_64_t0.01_cfl0.2`) produced by

```bash
OMP_NUM_THREADS=1 python -m benchmarks.riemann_2d_cks1998 --leg 1 32 --time 0.01 --cfl 0.2 \
    --output-dir validation/riemann2d_c2/smoke_ladder
```

and `metrics_full.json` is the merge of them by `merge_legs(...)`, which recomputes C2
(`self_convergence_l1`) from the stored cell-mean arrays rather than from any typed number.

* Measured C2 = 1.566362e-03 (tolerance 0.05, gate PASS).
* Steps: 29 (32x32) / 59 (64x64) -- same code path, same limiter, same CFL as the case.
* **This is not the case configuration**: the case runs t=0.25 on P1/P2; this ladder only proves
  the execution/evidence path works end to end at a cost a short test can afford.
* Production legs were queued as cj4312 (P1 32), cj4313 (P2 32), cj4314 (P1 64), cj4315 (P2 64)
  and had not started when this file was written, so the case's C2 stays `NOT_RUN`.
