import json
from benchmarks.forward_facing_step import gate_wall_reflection
r = gate_wall_reflection(ncells=(29,1), degree=1, wall_mode="solid",
                         x_wall=0.3, x_shock=0.25, xmax=0.35, ymax=0.5, extra_time=0.06)
print(json.dumps({k: r[k] for k in ("status","pass","plateau_cells","plateau_cells_available",
  "worst_relative_error_percent","plateau_mean_rho_error_percent","plateau_mean_p_error_percent",
  "l1_relative","linf_relative","steps","wall_time","final_time","reflected_rho_exact","reflected_p_exact")}, indent=1))
