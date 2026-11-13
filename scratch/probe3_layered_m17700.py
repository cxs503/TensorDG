import os, math
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch
from tensordg.elastic_1d_layered import (solve_elastic_1d_layered,
    rightgoing_pulse_initial_state, layered_stress_profile, layered_energy,
    exact_rightgoing_scattering)
torch.set_num_threads(1)

N, p, L, iface = 400, 2, 4.0, 2.0
n1 = 200
rho2, e2 = 4.0, 4.0
rho = [1.0]*n1 + [rho2]*(N-n1)
ey = [1.0]*n1 + [e2]*(N-n1)
v_init, s_init = rightgoing_pulse_initial_state(iface, 1.0, 1.0, 1.0, 0.05)
for t in (0.0, 0.5, 0.9, 1.1, 1.4):
    x, cf, meta = solve_elastic_1d_layered(ncells=N, degree=p, final_time=t,
                                           length=L, rho=rho, young=ey,
                                           bc="fixed-free",
                                           initial_velocity=v_init,
                                           initial_stress=s_init)
    xs, vs, ss = layered_stress_profile(x, cf, 8)
    i = int(torch.argmax(ss.abs()))
    ve, se = exact_rightgoing_scattering(xs, t, iface, L, 1.0, 1.0, rho2, e2, 1.0, 0.05,
                                        strict=t <= 2.0)
    ie = int(torch.argmax(se.abs()))
    l2 = float(torch.sqrt(torch.sum((ss-se)**2))/torch.sqrt(torch.sum(se**2)))
    print(f"t={t}: num max|s|={float(ss.abs().max()):.6f} at x={float(xs[i]):.4f} "
          f"(s={float(ss[i]):+.5f}) | exact max={float(se.abs().max()):.6f} at "
          f"x={float(xs[ie]):.4f} (s={float(se[ie]):+.5f}) | rel L2={l2:.3%} | "
          f"E={layered_energy(cf, L/N, rho, ey):.6e}")