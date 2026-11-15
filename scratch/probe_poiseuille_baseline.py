"""Quick probe: separate forcing-order and wall-convention contributions."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "TensorLBM" / "src"))
from tensorlbm.poiseuille_baseline import PoiseuilleConfig, run  # noqa: E402

variants = [
    ("legacy: simple forcing + node-centred reference", dict(forcing="simple", wall_origin=0.0)),
    ("simple forcing + halfway reference", dict(forcing="simple", wall_origin=0.5)),
    ("guo forcing + node-centred reference", dict(forcing="guo", wall_origin=0.0)),
    ("guo forcing + halfway reference (baseline)", dict(forcing="guo", wall_origin=0.5)),
]

out = {}
for label, kw in variants:
    cfg = PoiseuilleConfig(**kw, max_steps=200000, residual_tol=1e-12, check_every=200)
    t0 = time.time()
    res = run(cfg)
    dt = time.time() - t0
    out[label] = {
        "rel_l2": res["errors"]["rel_l2"],
        "rel_linf": res["errors"]["rel_linf"],
        "rel_u_max": res["errors"]["rel_u_max"],
        "steps": res["convergence"]["steps"],
        "residual": res["convergence"]["final_residual"],
        "converged": res["convergence"]["converged"],
        "mass_drift": res["mass_drift"],
        "wall_s": round(dt, 2),
    }
    print(f"{label:52s} relL2={res['errors']['rel_l2']:.4e} "
          f"relLinf={res['errors']['rel_linf']:.4e} steps={res['convergence']['steps']} "
          f"res={res['convergence']['final_residual']:.2e} t={dt:.1f}s")

# also the fixed-step legacy number (2000 steps, as delivered)
cfg = PoiseuilleConfig(forcing="simple", wall_origin=0.0, max_steps=2000, check_every=2000,
                       residual_tol=0.0)
res = run(cfg)
out["legacy at 2000 fixed steps"] = {"rel_l2": res["errors"]["rel_l2"], "steps": 2000}
print("legacy @2000 steps relL2 =", res["errors"]["rel_l2"])

Path("/tmp/probe_poiseuille.json").write_text(json.dumps(out, indent=2))