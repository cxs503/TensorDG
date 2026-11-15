import json, sys, time
from benchmarks.forward_facing_step import gate_wall_reflection
nx = int(sys.argv[1]); p = int(sys.argv[2]); mode = sys.argv[3]
t0 = time.time()
r = gate_wall_reflection(ncells=(nx,1), degree=p, wall_mode=mode, x_wall=1.0,
                         x_shock=0.4, xmax=1.2, extra_time=0.08)
print(json.dumps({k: r[k] for k in ("status","pass","plateau_cells",
  "worst_relative_error_percent","plateau_mean_rho_error_percent",
  "plateau_mean_p_error_percent","l1_relative","linf_relative","steps")} | {"seconds": round(time.time()-t0,1)}))
