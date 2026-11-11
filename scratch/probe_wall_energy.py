import torch, tensordg.ns_dg_2d as ns
import importlib.util, sys
spec = importlib.util.spec_from_file_location("t", "tests/test_ns_heat_conduction.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for ny in (8,16,32,64,128):
    r = m._wall_residual(ny)
    print(ny, ["%.3e"%float(r[...,k].abs().max()) for k in range(4)],
          "wallE=%.4e"%float(r[...,3][:,-1].abs().max()),
          "interiorE=%.4e"%float(r[...,3][:,1:-1].abs().max()))
