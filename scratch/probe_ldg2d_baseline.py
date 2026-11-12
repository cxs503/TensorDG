"""Baseline probe: current 2-D LDG prototype (mirror closure) convergence."""
import numpy as np
from tensordg.poisson_2d_ldg import solve_poisson_2d_ldg
from tensordg.convergence import observed_order

for k in (1, 2):
    errs = []
    grids = [4, 8, 16]
    for n in grids:
        try:
            r = solve_poisson_2d_ldg(n, k)
            errs.append(r["l2_relative_error"])
            print(f"k={k} N={n:3d} ndofs={r['ndofs']:6d} relL2={r['l2_relative_error']:.6e} grad={r['grad_relative_error']:.3e}")
        except Exception as exc:
            errs.append(float("nan"))
            print(f"k={k} N={n} FAILED: {exc}")
    orders = [observed_order(errs[i], errs[i+1], grids[i], grids[i+1]) for i in range(len(grids)-1)]
    print(f"  k={k} orders={orders}")