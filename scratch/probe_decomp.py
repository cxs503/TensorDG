import torch, importlib.util, tensordg.ns_dg_2d as ns
spec = importlib.util.spec_from_file_location("t", "tests/test_ns_heat_conduction.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

# original and capture
orig = ns._boundary_flux
cap = {}
def cap_bf(side, wall, axis, h, mu, kappa, penalty, normal_scale, tangential_scale, **kw):
    out = orig(side, wall, axis, h, mu, kappa, penalty, normal_scale, tangential_scale, **kw)
    cap[(axis, kw.get("side_name"))] = (out, kw, penalty, normal_scale)
    return out
ns._boundary_flux = cap_bf
r = m._wall_residual(32)
ns._boundary_flux = orig
for k, (out, kw, pen, nsc) in cap.items():
    print(k, "penalty=%.3f normal_scale=%.3f" % (pen, nsc),
          "fluxE(last/mean)=%.6f" % float(out[...,3].abs().max()))
    print("   energy flux values:", out[..., 3].flatten()[:3].tolist())
