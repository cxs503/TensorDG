import torch, importlib.util
spec = importlib.util.spec_from_file_location("t", "tests/test_ns_heat_conduction.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import tensordg.ns_dg_2d as ns
for p in (0.5, 1.0, 2.0):
    m.PRESSURE = p
    r = m._wall_residual(32)[...,3]
    print("p=%.1f topwall m0=%.4e botwall m0=%.4e"%(p, float(r[:,-1,0].abs().max()), float(r[:,0,0].abs().max())))
m.PRESSURE = 1.0
for k in (0.5, 1.0, 2.0):
    m.KAPPA = k * (m.MU * ns.CP / m.PRANDTL)
    r = m._wall_residual(32)[...,3]
    print("kappa x%.1f top=%.4e bot=%.4e"%(k, float(r[:,-1,0].abs().max()), float(r[:,0,0].abs().max())))
