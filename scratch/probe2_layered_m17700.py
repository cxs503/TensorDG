import os, math
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch
from tensordg.elastic_1d_layered import (solve_elastic_1d_layered,
    rightgoing_pulse_initial_state, layered_stress_profile, layered_energy,
    exact_rightgoing_scattering)
from tensordg.elastic_1d import solve_elastic_1d, gaussian_pulse
torch.set_num_threads(1)

N, p, L, iface = 400, 2, 4.0, 2.0
n1 = 200
rho = [1.0]*n1 + [4.0]*(N-n1)
ey = [1.0]*n1 + [4.0]*(N-n1)
v_init, s_init = rightgoing_pulse_initial_state(iface, 1.0, 1.0, 1.0, 0.05)
for t in (0.0, 0.5, 1.0, 1.4):
    x, cf, meta = solve_elastic_1d_layered(ncells=N, degree=p, final_time=t,
                                           length=L, rho=rho, young=ey,
                                           bc="fixed-free",
                                           initial_velocity=v_init,
                                           initial_stress=s_init)
    xs, vs, ss = layered_stress_profile(x, cf, 8)
    i = int(torch.argmax(ss.abs()))
    i0 = int(torch.argmax(vs.abs()))
    print(f"t={t}: max|s|={float(ss.abs().max()):.6f} at x={float(xs[i]):.4f} "
          f"(sigma={float(ss[i]):+.6f}); max|v|={float(vs.abs().max()):.6f} at "
          f"x={float(xs[i0]):.4f}; energy={layered_energy(cf, L/N, rho, ey):.6e}")
# compare with a pure homogeneous run of same initial data
print("--- homogeneous reference (should move right at c=1)")
x, cf, meta = solve_elastic_1d(ncells=N, degree=p, final_time=1.0, length=L,
                               bc="fixed-free", initial_velocity=v_init,
                               initial_stress=s_init)
xs, vs, ss = layered_stress_profile(x, cf, 8)
i = int(torch.argmax(ss.abs()))
print(f"t=1.0 homog: max|s|={float(ss.abs().max()):.6f} at x={float(xs[i]):.4f}")