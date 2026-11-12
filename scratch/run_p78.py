import json, time
from benchmarks.p_refinement_exponential import temporal_converged_advection, NCELLS
from tensordg.advection_1d import solve_advection, l2_error
out = {}
for p in (7, 8):
    t0 = time.time()
    r = temporal_converged_advection(NCELLS, p)
    out[p] = {k: r[k] for k in ("relative_l2", "cfl", "steps", "dt", "temporal_change",
                                "temporal_converged", "refinements")}
    out[p]["seconds"] = time.time() - t0
    print(p, out[p], flush=True)
json.dump({str(k): v for k, v in out.items()}, open("scratch/p78.json", "w"), indent=1)
