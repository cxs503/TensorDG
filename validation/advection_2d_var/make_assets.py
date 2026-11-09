"""Produce validation/advection_2d_var/results.json + the audit figure."""
import json, math, os, sys, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tensordg.advection_2d_var import (DGAdvection2D, gaussian_bump, reentrant_vortex,
                                       rotated_profile, solid_body_rotation, trig_profile)

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.abspath(os.path.join(OUT, "..", "..", "docs", "reports", "figs",
                                   "tensordg_advection_2d_var_reference_audit.png"))
def eoc(e): return [math.log(e[i]/e[i+1])/math.log(2) for i in range(len(e)-1)]

def vortex(p, meshes):
    errs = []
    for n in meshes:
        s = DGAdvection2D(degree=p, ncells=n)
        c = s.project(trig_profile)
        r = s.solve(c, reentrant_vortex(period=1.0), final_time=1.0, cfl=0.3)
        errs.append(s.l2_error(r["coeff"], trig_profile, t=1.0)["relative_l2"])
        print("vortex", p, n, errs[-1], flush=True)
    return {"degree": p, "meshes": list(meshes), "errors": errs, "eoc": eoc(errs)}

def rotation(p, meshes):
    rows = []
    vel = solid_body_rotation(omega=2*math.pi); ex = rotated_profile(gaussian_bump, 2*math.pi)
    for n in meshes:
        s = DGAdvection2D(degree=p, ncells=n)
        c = s.project(gaussian_bump)
        r = s.solve(c, vel, final_time=0.25, cfl=0.3)
        d = s.l2_error(r["coeff"], ex, t=0.25, radius=0.25)["relative_l2"]
        g = s.l2_error(r["coeff"], ex, t=0.25)["relative_l2"]
        rows.append({"n": n, "disk_relative_l2": d, "global_relative_l2": g})
        print("rotation", p, n, d, g, flush=True)
    return {"degree": p, "rows": rows}

def rigid_global_trig(meshes=(8, 16)):
    vel = solid_body_rotation(omega=2*math.pi); ex = rotated_profile(trig_profile, 2*math.pi)
    errs = []
    for n in meshes:
        s = DGAdvection2D(degree=2, ncells=n)
        c = s.project(trig_profile)
        r = s.solve(c, vel, final_time=0.25, cfl=0.3)
        errs.append(s.l2_error(r["coeff"], ex, t=0.25)["relative_l2"])
        print("rigid-global-trig", n, errs[-1], flush=True)
    return {"meshes": list(meshes), "errors": errs}

payload = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "torch": torch.__version__,
           "vortex": [vortex(1, (8, 16, 32)), vortex(2, (8, 16, 32)), vortex(3, (8, 16))],
           "rotation": [rotation(1, (16, 32)), rotation(2, (16, 32))],
           "rigid_global_trig": rigid_global_trig()}
json.dump(payload, open(os.path.join(OUT, "results.json"), "w"), indent=1)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
lbl = {1: "P1 (design 2)", 2: "P2 (design 3)", 3: "P3 (design 4)"}
for fam in payload["vortex"]:
    p = fam["degree"]
    a1.loglog(fam["meshes"], fam["errors"], "o-", label="%s, order %.2f" % (lbl[p], fam["eoc"][-1]))
    n0 = fam["meshes"][0]; e0 = fam["errors"][0]
    ref = [e0 * (n0 / n) ** (p + 1) for n in fam["meshes"]]
    a1.loglog(fam["meshes"], ref, "k--", lw=0.8, alpha=0.5)
a1.set_xlabel("n cells per direction"); a1.set_ylabel("relative L2 error")
a1.set_title("(a) reentrant vortex: valid global gate\n(dashed: design slope p+1)")
a1.legend(fontsize=8); a1.grid(True, which="both", alpha=0.3)
for fam in payload["rotation"]:
    ns = [r["n"] for r in fam["rows"]]
    a2.loglog(ns, [r["disk_relative_l2"] for r in fam["rows"]], "o-",
              label="disk norm P%d" % fam["degree"])
    a2.loglog(ns, [r["global_relative_l2"] for r in fam["rows"]], "s--",
              label="global norm P%d" % fam["degree"])
a2.axhline(0.03, color="r", lw=1, ls=":", label="3% gate")
a2.set_xlabel("n cells per direction"); a2.set_ylabel("relative L2 error")
a2.set_title("(b) rigid rotation: seam-contaminated global norm\nvs well-posed disk norm r<=0.25")
a2.legend(fontsize=8); a2.grid(True, which="both", alpha=0.3)
fig.tight_layout(); fig.savefig(FIG, dpi=140)
print("wrote", FIG)
