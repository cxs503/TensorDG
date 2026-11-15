"""Probe: does the force-splitting order remove the constant offset?

Variants (all BGK + Guo terms + halfway bounce-back, Ny=16 channel):
  a  collide + F_i, then bounce-back        (current baseline)
  b  collide, bounce-back, then + F_i       (force after streaming)
  c  collide + F_i/2, bounce-back, + F_i/2  (split / half-force)
  d  collide + F_i/2, bounce-back, + F_i/2 with u_shift inside F_i only
"""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "TensorLBM" / "src"))
from tensorlbm.lattice import D2Q9  # noqa: E402
from tensorlbm.boundaries import HalfwayBounceBack  # noqa: E402
from tensorlbm.poiseuille_baseline import (  # noqa: E402
    PoiseuilleConfig, analytic_profile, guo_forcing, solid_mask, macroscopic_with_force,
)

Q = D2Q9.Q
F = 1e-5


def step_variant(f, walls, omega, variant):
    rho, ux, uy = macroscopic_with_force(f, F)
    feq = D2Q9.equilibrium(rho, ux, uy)
    coll = f - omega * (f - feq)
    src = guo_forcing(ux, uy, F, omega)
    if variant == "a":
        return walls.stream(coll + src)
    if variant == "b":
        return walls.stream(coll) + src
    if variant == "c":
        return walls.stream(coll + 0.5 * src) + 0.5 * src
    raise ValueError(variant)


def run_variant(variant, tau, Ny=16, Nx=4, tol=1e-11, max_steps=40000, every=100):
    cfg = PoiseuilleConfig(Ny=Ny, Nx=Nx, tau=tau, init="analytic")
    rho = torch.ones((Ny, Nx), dtype=torch.float64)
    z = torch.zeros((Ny, Nx), dtype=torch.float64)
    u0 = analytic_profile(cfg).view(Ny, 1).expand(Ny, Nx).contiguous()
    f = D2Q9.equilibrium(rho, u0, z)
    walls = HalfwayBounceBack(solid_mask(Ny, Nx))
    omega = 1.0 / tau
    u_prev = None
    steps = 0
    while steps < max_steps:
        for _ in range(every):
            f = step_variant(f, walls, omega, variant)
        steps += every
        _, ux, _ = macroscopic_with_force(f, F)
        if u_prev is not None and float((ux - u_prev).abs().max()) / cfg.u_max_analytic < tol:
            break
        u_prev = ux
    _, ux, _ = macroscopic_with_force(f, F)
    prof = ux[:, Nx // 2]
    ana = analytic_profile(cfg)
    d = (prof - ana)[1:-1]
    return {
        "offset_over_F": float(d.mean()) / F,
        "rel_l2": float(torch.sqrt(torch.mean(d ** 2))) / cfg.u_max_analytic,
        "rel_linf": float(d.abs().max()) / cfg.u_max_analytic,
        "steps": steps,
    }


if __name__ == "__main__":
    for tau in (0.6, 0.8, 1.0, 1.2, 1.5):
        row = []
        for v in ("a", "b", "c"):
            r = run_variant(v, tau)
            row.append(f"{v}: off/F={r['offset_over_F']:+.4f} relL2={r['rel_l2']:.2e}")
        print(f"tau={tau:4.2f}  " + "   ".join(row), flush=True)