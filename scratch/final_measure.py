import json, math, warnings, numpy as np, torch
warnings.filterwarnings("ignore")
from tensordg.euler_curved import (CurvilinearEuler, warped_box_mesh, straight_box_mesh,
                                   hllc_normal, euler_flux, conserved_from_primitive)
from tensordg.euler_2d import vortex_primitive
L, S, A, G = 10.0, 5.0, 0.06, 3
def prim(t):
    def f(x,y):
        r,u,v,p = vortex_primitive(torch.as_tensor(np.ascontiguousarray(x)),
                                   torch.as_tensor(np.ascontiguousarray(y)),
                                   time=t, strength=S, background_velocity=(1.0,1.0), period=L)
        return r.numpy(),u.numpy(),v.numpy(),p.numpy()
    return f
out = {"free_stream": [], "convergence": [], "conservation": {}, "hllc": {}}
for p in (1,2):
    m = warped_box_mesh(4,4,G,length=L,amplitude=A)
    good = CurvilinearEuler(m, degree=p).free_stream_defect(steps=20)
    bad = CurvilinearEuler(m, degree=p, volume_derivative_degree=p).free_stream_defect(steps=0)
    out["free_stream"].append({"degree":p,"control":good["relative"],"drift":good["drift"],
        "metric_identity":good["metric"],"negative_control":bad["relative"],
        "delay_ratio":bad["relative"]/max(good["relative"],1e-300)})
for family, mk in (("warped", lambda n: warped_box_mesh(n,n,G,length=L,amplitude=A)),
                   ("straight", lambda n: straight_box_mesh(n,n,1,length=L))):
    errs = []
    for n in (8,16):
        s = CurvilinearEuler(mk(n), degree=1, cfl=0.2)
        st = s.project(prim(0.0)); st, dt, steps = s.evolve(st, 0.1)
        e = s.relative_errors(st, prim(0.1)); errs.append(e["worst"])
        out["convergence"].append({"family":family,"degree":1,"cells":n,"steps":steps,
                                   "dt":dt,"error":e["worst"]})
    out["convergence"][-1]["observed_order"] = None if errs[-1]<=0 else math.log(errs[0]/errs[-1])/math.log(2)
s = CurvilinearEuler(warped_box_mesh(8,8,G,length=L,amplitude=A), degree=1, cfl=0.2)
st = s.project(prim(0.0)); before = s.totals(st); st, dt, steps = s.evolve(st, 0.5); after = s.totals(st)
out["conservation"] = {"steps":steps,"dt":dt,
    "relative_drift":float(np.max(np.abs(after-before))/np.max(np.abs(before)))}
state = conserved_from_primitive(np.array([1.0]), np.array([0.3]), np.array([-0.2]), np.array([1.0]))
f = hllc_normal(state, state, np.array([0.6]), np.array([0.8])); fx, fy = euler_flux(state)
out["hllc"] = {"consistency": float(np.max(np.abs(f-(fx*0.6+fy*0.8))))}
import pathlib
pathlib.Path("validation/euler_curved").mkdir(parents=True, exist_ok=True)
pathlib.Path("validation/euler_curved/metrics_m21360.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
