import torch, importlib.util
spec = importlib.util.spec_from_file_location("t", "tests/test_ns_heat_conduction.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for ny in (8,16,32,64,128):
    r = m._wall_residual(ny)[...,3]
    print("ny=%3d"%ny, "topwall mode0=%.3e mode1=%.3e"%(
        float(r[:,-1,0].abs().max()), float(r[:,-1,1].abs().max())),
        "| botwall m0=%.3e m1=%.3e"%(float(r[:,0,0].abs().max()), float(r[:,0,1].abs().max())))
