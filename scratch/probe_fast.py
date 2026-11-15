import json, sys, time
from benchmarks.forward_facing_step import gate_wall_reflection
for nx,p,mode in ((40,1,"boundary"),(40,1,"solid")):
    t0=time.time()
    r = gate_wall_reflection(ncells=(nx,1), degree=p, wall_mode=mode, x_wall=0.6,
                             x_shock=0.2, xmax=0.7, extra_time=0.1)
    print(mode, json.dumps({k: r[k] for k in ("status","pass","plateau_cells",
      "worst_relative_error_percent","plateau_mean_rho_error_percent",
      "plateau_mean_p_error_percent","l1_relative","steps")}), f"{time.time()-t0:.1f}s", flush=True)
