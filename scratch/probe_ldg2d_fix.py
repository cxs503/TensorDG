import numpy as np, math
from tensordg.poisson_2d_ldg import solve_poisson_2d_ldg

grids = [4, 8, 16, 32]
for closure in ("flux", "mirror"):
    for k in (1, 2):
        errs = []
        for n in grids:
            try:
                r = solve_poisson_2d_ldg(n, k, boundary_closure=closure)
                errs.append(r["l2_relative_error"])
                print(f"{closure:6s} k={k} N={n:3d} ndofs={r['ndofs']:6d} relL2={r['l2_relative_error']:.4e} grad={r['grad_relative_error']:.3e}")
            except Exception as exc:
                errs.append(float("nan")); print(f"{closure} k={k} N={n} FAIL {exc}")
        orders = [math.log(errs[i]/errs[i+1])/math.log(grids[i+1]/grids[i]) for i in range(len(errs)-1)]
        print(f"   {closure} k={k} local orders={['%.3f'%o for o in orders]}")
