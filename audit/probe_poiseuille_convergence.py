#!/usr/bin/env python3
"""Independent reproduction probe for m21003 (XVAL-LBM-MASS-EXCHANGE-SELECTOR).

Runs the audited mission's own BGK Poiseuille configuration (Nx=64, Ny=32,
tau=0.8, F=1e-5) but continues the run past the delivered n_steps=2000 to test
whether the delivered ~11% error is an implementation error or an
under-converged benchmark configuration.

Force term and streaming are taken unchanged from the mission's
benchmarks/poiseuille.py (stream_with_bounceback) - the only change is the
step count and that the profile is kept for plotting.

Writes audit/artifacts/poiseuille_convergence.json
"""
import sys, json, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
REF = ROOT / "refs" / "TensorLBM_m21003"
sys.path.insert(0, str(REF / "src"))
sys.path.insert(0, str(REF / "benchmarks"))

from tensorlbm import D2Q9, MassExchangeSelector, Scheme
import poiseuille as P

Q = D2Q9.Q
Nx, Ny, F, tau = 64, 32, 1e-5, 0.8
omega = 1.0 / tau
nu = D2Q9.cs2.item() * (tau - 0.5)
H = Ny - 1
u_max_ana = F * H ** 2 / (8 * nu)
fz = (D2Q9.w * 3.0 * D2Q9.e[:, 0]).view(Q, 1, 1) * F


def profile(f):
    _, ux, _ = D2Q9.macroscopic(f)
    return ux[:, Nx // 2]


def rel_l2(prof, shift=0.0, width=None):
    """rel L2 vs analytic parabola; shift = wall position convention."""
    y = torch.arange(Ny, dtype=torch.float64)
    if width is None:                      # node-centred: zero at y=0 and y=H
        ua = (F / (2 * nu)) * y * (H - y)
    else:                                  # half-way bounce-back walls
        ua = (F / (2 * nu)) * (y - 0.5) * (width + 0.5 - y)
    sl = slice(1, -1)
    e = torch.sqrt(torch.mean((prof[sl] - ua[sl]) ** 2)).item()
    return e / u_max_ana


def main():
    rho = torch.ones(Ny, Nx, dtype=torch.float64)
    z = torch.zeros(Ny, Nx, dtype=torch.float64)
    f = D2Q9.equilibrium(rho, z, z)
    sel = MassExchangeSelector(scheme=Scheme.BGK, omega=omega)
    marks = [2000, 5000, 10000, 16000]
    out = {"case": {"Nx": Nx, "Ny": Ny, "tau": tau, "F": F, "nu": nu,
                    "H": H, "u_max_analytic": u_max_ana},
           "checkpoints": [], "elapsed_s": None}
    t0, step = time.time(), 0
    for m in marks:
        for _ in range(m - step):
            r, ux, uy = D2Q9.macroscopic(f)
            f = P.stream_with_bounceback(sel.collide(f, r, ux, uy) + fz, Ny, Nx)
        step = m
        p = profile(f)
        out["checkpoints"].append({
            "steps": m,
            "rel_l2_node_centred": rel_l2(p),
            "rel_l2_halfway_bb": rel_l2(p, width=Ny - 2),
            "u_max_numeric": p.max().item(),
            "u_max_ratio": p.max().item() / u_max_ana,
            "profile": p.tolist(),
        })
        print(json.dumps({k: v for k, v in out["checkpoints"][-1].items()
                          if k != "profile"}), flush=True)
    out["elapsed_s"] = time.time() - t0
    (ROOT / "artifacts" / "poiseuille_convergence.json").write_text(
        json.dumps(out, indent=2))
    print("elapsed", round(out["elapsed_s"], 1), "s")


if __name__ == "__main__":
    main()
