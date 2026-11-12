import math, torch
from tensordg.elastic_1d import solve_elastic_1d, initial_pulse_velocity, dalembert_pulse
from tensordg.elastic_2d import ElasticMedium, solve_elastic_2d
med2 = ElasticMedium(rho=1.0, lam=0.5, mu=0.25)   # c_p = sqrt((lam+2mu)/rho) = 1
print("cp", med2.cp)
def init(x, y):
    out = torch.zeros(*x.shape, 5, dtype=torch.float64)
    out[..., 0] = torch.exp(-((x - 0.5) / 0.05) ** 2)
    return out
T = 0.2
for n in (50, 100, 200):
    res2 = solve_elastic_2d(init, nx=n, ny=1, degree=2, final_time=T, cfl=0.3, medium=med2)
    xg, yg = res2.quadrature_grid()
    centers, c1, meta = solve_elastic_1d(ncells=n, degree=2, final_time=T, cfl=0.3, bc="periodic",
                                         rho=1.0, young=1.0, initial_velocity=initial_pulse_velocity,
                                         initial_stress=lambda x: torch.zeros_like(x))
    vref, sref = dalembert_pulse(centers, T, rho=1.0, young=1.0, center=0.5, width=0.05)
    v1 = c1[:, 0, 0]
    print(f"n={n} 1D solver vs d'Alembert: v rel={((v1-vref).norm()/vref.norm()).item():.3e}")
    x2 = xg[:, 0, :, 0].reshape(-1)
    v2 = res2.fields()[:, 0, :, 0, 0].reshape(-1)
    ve, _ = dalembert_pulse(x2, T, rho=1.0, young=1.0, center=0.5, width=0.05)
    print(f"      2D solver vs d'Alembert: v rel={((v2-ve).norm()/ve.norm()).item():.3e}")
