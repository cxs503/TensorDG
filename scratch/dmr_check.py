import time
import numpy as np
from tensordg.euler_dg_2d import cell_fields
from benchmarks.double_mach_reflection import (
    POST_SHOCK, PRE_SHOCK, SHOCK_SPEED, rh_residuals, solve_double_mach,
    measure_primary_front, analytic_front_x, sample_behind_front, NORMAL)

print("POST_SHOCK", POST_SHOCK)
print("RH residuals", {k: float(v) for k, v in rh_residuals(PRE_SHOCK, POST_SHOCK).items()})

for t in (0.01, 0.02):
    t0 = time.time()
    x, y, coeff, meta = solve_double_mach(ncells=(96, 24), degree=1,
                                          final_time=t, cfl=0.15)
    rho = cell_fields(coeff)["density"].cpu().numpy()
    front = measure_primary_front(rho, x, y)
    ana = {k: float(analytic_front_x(k, t)) for k in front}
    print(f"t={t} steps={meta['steps']} wall={time.time()-t0:.1f}s "
          f"rho=({rho.min():.3f},{rho.max():.3f})")
    for k in sorted(front):
        print(f"   y={k:.2f} x={front[k]:.4f} exact={ana[k]:.4f} err={front[k]-ana[k]:+.4f}")