import torch
from tests.test_g3_embedded_wall_equivalence import _run, _reference, NX_FLUID, DX, FINAL_TIME
wr = _reference()
cb, mb = _run("boundary"); cs, ms = _run("solid")
scale = cb.abs().max().item()
d = (cb - cs).abs().max().item()
print(f"wall_time={wr.wall_time:.5f} final_time={FINAL_TIME} steps={mb['steps']}")
print(f"max|boundary-solid|={d:.3e} scale={scale:.3e} relative={d/scale:.3e}")
print("steps_equal", ms["steps"] == mb["steps"])
x = (torch.arange(NX_FLUID, dtype=torch.float64)+0.5)*DX
rho = cs[..., 0, 0, 0][:, 0]
front = wr.front(FINAL_TIME)
print(f"front={front:.4f} wall={wr.x_wall} post_rho={wr.post[0]:.4f} reflected_rho={wr.reflected[0]:.4f}")
for i in range(NX_FLUID):
    print(f"  x={x[i]:.3f} rho={rho[i].item():.4f}")
print("mass_res", mb["conservation_residual_total"]["mass"]["relative"], ms["conservation_residual_total"]["mass"]["relative"])
