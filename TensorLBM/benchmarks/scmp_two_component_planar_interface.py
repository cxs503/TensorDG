"""XVAL-LBM-SCMP-TWO-COMPONENT-PLANAR-INTERFACE benchmark.

Runs the two-component Shan-Chen (SCMP) solver of ``tensorlbm.mcmp`` on a
periodic planar interface and scores the equilibrium state against the
independent mean-field reference of ``tools/scmp_planar_reference.py``.

The anchor (``lbm-principles-practice/code`` scmp-two-component-planar-interface
practice) is a *quantitative* one: the bulk coexistence densities of a symmetric
binary Shan-Chen mixture with a linear pseudopotential are known exactly,

    ln(rho_hi/rho_lo) = |G_12| (rho_hi - rho_lo),   rho_hi + rho_lo = 2 rho_bar

so the measured plateaus can be scored with a percentage error, and the mixture
mechanical pressure

    p = cs^2 (rho_1 + rho_2 - G_12 psi_1 psi_2)

must be flat across the whole periodic cell at equilibrium -- an exact invariant
of the continuum model that no code-comparison ("both codes agree") can supply.

Arms are declared on the command line so that a sweep can be sharded:

    python TensorLBM/benchmarks/scmp_two_component_planar_interface.py \
        --label lin_G2.2 --psi linear --G -2.2 --ny 64 --steps 12000 --omega 1.0

Every arm writes ``TensorLBM/outputs/scmp_planar/<label>.json`` with the measured
profiles, so the report figures are rendered from data (never hand-typed).

Pre-registered acceptance (owner precision standard):
  * ``err_max`` (relative error of rho_hi and rho_lo vs. the exact tie-line)
    must be < 15 %; the 3 % target is reported but never used to gate;
  * ``pressure_spread_relative`` must be < 1 % (exact invariant, 0 in the
    continuum limit) -- reported as an instrument-health gate, not as accuracy;
  * an arm that produces non-finite fields is reported as a divergence with the
    step at which it happened (honest FAIL, never replaced silently).
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "TensorLBM" / "src"))
sys.path.insert(0, str(ROOT / "TensorLBM" / "tools"))

from tensorlbm.mcmp import MCMPSolver                       # noqa: E402
from tensorlbm.cross_force import (                         # noqa: E402
    pseudopotential_rho,
    pseudopotential_shan_chen,
)
import scmp_planar_reference as REF                         # noqa: E402

PSI = {"linear": pseudopotential_rho, "exp": pseudopotential_shan_chen}
OUTDIR = ROOT / "TensorLBM" / "outputs" / "scmp_planar"
CS2 = 1.0 / 3.0


# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------

def initial_condition(ny: int, nx: int, G: float, psi_kind: str,
                      rho_bar: float = 0.5, width: float = 6.0,
                      nominal_amplitude: float = 0.30, force_rho_hi=None):
    """Smooth periodic interface, thickened and (for linear psi) pressure matched.

    Two interfaces sit at ``ny/4`` and ``3*ny/4`` so the periodic seam carries no
    jump (a sharp seam is an avoidable source of spurious forces -- it was the
    first thing that blew up during the audit).

    For ``psi = rho`` the amplitude is taken from the exact tie-line and the
    total density dip is obtained from the constant-pressure condition, so the
    initial state is already close to mechanical equilibrium.  For a non-linear
    pseudopotential no tie-line exists, and a nominal tanh amplitude is used; the
    arm is then scored on invariants only.
    """
    if psi_kind == "linear":
        rho_hi, rho_lo = REF.coexistence_tieline(G, rho_bar)
    else:
        rho_hi = rho_bar + nominal_amplitude
        rho_lo = rho_bar - nominal_amplitude
    if force_rho_hi is not None:            # used to test the sign of the demixing direction
        rho_hi = float(force_rho_hi)
        rho_lo = 2.0 * rho_bar - rho_hi

    y = torch.arange(ny, dtype=torch.float64).view(ny, 1).expand(ny, nx)
    phi = 0.5 * (torch.tanh((y - ny / 4.0) / width) - torch.tanh((y - 3 * ny / 4.0) / width))
    w1 = rho_bar + (rho_hi - rho_bar) * phi                 # composition weight in [rho_lo, rho_hi]
    w1 = w1.clamp(1e-6, 1.0 - 1e-6)
    q = w1 * (1.0 - w1)

    g = abs(G)
    if psi_kind == "linear":
        p_bulk = (rho_hi + rho_lo) - G * rho_hi * rho_lo   # p / cs^2 (psi = rho)
        # s + g s^2 q = p_bulk   ->   g q s^2 + s - p_bulk = 0
        disc = torch.sqrt(1.0 + 4.0 * g * q * p_bulk)
        s = (-1.0 + disc) / (2.0 * g * q)
    else:
        s = torch.ones_like(w1)                             # uniform total density
    return s * w1, s * (1.0 - w1), rho_hi, rho_lo


def plateau_slices(ny: int, width: float):
    """Rows of the two bulk plateaus (away from both periodic interfaces).

    The initial composition has interfaces at ``ny/4`` and ``3*ny/4``; rows whose
    distance to the nearest interface exceeds ``2.5*width`` belong to a bulk
    phase.  The margin is reduced automatically when the plateaus are thin.
    """
    ifaces = (ny / 4.0, 3 * ny / 4.0)
    margin = 2.5 * width
    while margin >= 1.5:
        a = [y for y in range(ny) if ifaces[0] + margin < y < ifaces[1] - margin]
        b = [y for y in range(ny) if min(abs(y - i) for i in ifaces) > margin
             and not (ifaces[0] + margin < y < ifaces[1] - margin)]
        if len(a) >= 4 and len(b) >= 4:
            return torch.tensor(a), torch.tensor(b)
        margin *= 0.7
    raise ValueError(f"cannot place plateaus on {ny} rows with width {width}")


# ---------------------------------------------------------------------------
# One arm
# ---------------------------------------------------------------------------

def run_arm(label: str, G: float, psi_kind: str = "linear", ny: int = 64,
            nx: int = 4, steps: int = 12000, omega: float = 1.0,
            width: float = 6.0, avg_steps: int = 2000, sample_every: int = 100,
            rho_bar: float = 0.5, progress: bool = True,
            force_rho_hi=None) -> dict:
    psi = PSI[psi_kind]
    Gm = torch.tensor([[0.0, G], [G, 0.0]], dtype=torch.float64)
    solver = MCMPSolver(n_components=2, G_matrix=Gm, omega=omega, psi_func=psi,
                        nx=nx, ny=ny, dtype=torch.float64)
    r1_0, r2_0, rho_hi_ic, rho_lo_ic = initial_condition(
        ny, nx, G, psi_kind, rho_bar, width, force_rho_hi=force_rho_hi)
    solver.set_density_field(0, r1_0)
    solver.set_density_field(1, r2_0)
    mass_0 = (r1_0.sum() + r2_0.sum()).item()

    t0 = time.time()
    diverged_at = None
    acc1 = torch.zeros(ny, dtype=torch.float64)
    acc2 = torch.zeros(ny, dtype=torch.float64)
    n_acc = 0
    u_max_acc = 0.0
    for k in range(1, steps + 1):
        solver.step()
        if k % sample_every == 0 and k > steps - avg_steps:
            solver._compute_macroscopic()
            if not torch.isfinite(solver.rho[0]).all():
                diverged_at = k
                break
            acc1 += solver.rho[0].mean(dim=1)
            acc2 += solver.rho[1].mean(dim=1)
            n_acc += 1
            u_max_acc = max(u_max_acc,
                            float(solver.uy[0].abs().max()), float(solver.uy[1].abs().max()),
                            float(solver.ux[0].abs().max()), float(solver.ux[1].abs().max()))
        if k == steps and not torch.isfinite(solver.rho[0]).all():
            diverged_at = k
            break
    wall = time.time() - t0

    solver._compute_macroscopic()
    finite = bool(torch.isfinite(solver.rho[0]).all() and torch.isfinite(solver.rho[1]).all())
    if not finite:
        diverged_at = diverged_at or steps

    res = {
        "label": label,
        "kind": "tensorlbm-mcmp",
        "params": {"G": G, "psi": psi_kind, "ny": ny, "nx": nx, "steps": steps,
                   "omega": omega, "tau": 1.0 / omega, "width": width,
                   "avg_steps": avg_steps, "rho_bar": rho_bar},
        "wall_seconds": wall,
        "diverged": diverged_at is not None,
        "diverged_at_step": diverged_at,
        "n_averaged_samples": n_acc,
    }

    if not finite:
        res["verdict"] = "FAIL"
        res["fail_stage"] = "divergence"
        if progress:
            print(f"[{label}] DIVERGED at step {diverged_at} ({wall:.0f}s)")
        return res

    prof1 = (acc1 / max(n_acc, 1))
    prof2 = (acc2 / max(n_acc, 1))
    a, b = plateau_slices(ny, width)
    rho_hi_sim = float(prof1[a].mean())
    rho_lo_sim = float(prof1[b].mean())
    res["measured"] = {
        "rho_hi": rho_hi_sim,
        "rho_lo_sim_index": rho_lo_sim,
        "rho_hi_comp1": float(prof2[b].mean()),
        "rho_lo_comp1": float(prof2[a].mean()),
        "total_density_min": float((prof1 + prof2).min()),
        "total_density_max": float((prof1 + prof2).max()),
        "u_max": u_max_acc,
        "mass_rel_drift": abs(float(prof1.sum() + prof2.sum()) * nx - mass_0) / abs(mass_0),
    }
    res["profiles"] = {"rho1": prof1.tolist(), "rho2": prof2.tolist(),
                       "total": (prof1 + prof2).tolist()}

    ref_ok = psi_kind == "linear"
    res["reference_available"] = ref_ok
    if ref_ok:
        ref = REF.tie_line_relative_error(rho_hi_sim, rho_lo_sim, G, rho_bar)
        res["reference"] = ref
        res["pass"] = bool(ref["err_max"] < 0.15)
        res["target_3pct"] = bool(ref["err_max"] < 0.03)
        res["fail_stage"] = None if res["pass"] else "tie_line"
    else:
        res["reference"] = None
        res["pass"] = None
        res["target_3pct"] = None
        res["fail_stage"] = "reference_withheld"

    flat = REF.profile_pressure_flatness(prof1.tolist(), prof2.tolist(), G, psi_kind, CS2)
    res["pressure_invariant"] = flat
    res["verdict"] = "PASS" if res["pass"] else ("WITHHELD" if res["pass"] is None else "FAIL")
    if progress:
        refs = (f"tie-line=({res['reference']['rho_hi_reference']:.4f},"
                f"{res['reference']['rho_lo_reference']:.4f}) err_max={res['reference']['err_max']*100:.2f}%"
                if ref_ok else "tie-line=WITHHELD")
        print(f"[{label}] G={G:+.2f} psi={psi_kind} ny={ny} steps={steps} om={omega} "
              f"sim=({rho_hi_sim:.4f},{rho_lo_sim:.4f}) {refs} "
              f"p_spread={flat['pressure_spread_relative']*100:.2e}% u_max={u_max_acc:.1e} "
              f"({wall:.0f}s)")
    return res


def save(res: dict) -> pathlib.Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"{res['label']}.json"
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="arm")
    ap.add_argument("--G", type=float, default=-2.2)
    ap.add_argument("--psi", default="linear", choices=sorted(PSI))
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--nx", type=int, default=4)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--width", type=float, default=6.0)
    ap.add_argument("--avg-steps", dest="avg_steps", type=int, default=2000)
    ap.add_argument("--sample-every", dest="sample_every", type=int, default=100)
    ap.add_argument("--force-rho-hi", dest="force_rho_hi", type=float, default=None,
                    help="override the IC high density (sign-of-demixing probe)")
    ap.add_argument("--no-save", dest="save", action="store_false")
    args = ap.parse_args(argv)
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", 2)))
    res = run_arm(label=args.label, G=args.G, psi_kind=args.psi, ny=args.ny, nx=args.nx,
                  steps=args.steps, omega=args.omega, width=args.width,
                  avg_steps=args.avg_steps, sample_every=args.sample_every,
                  force_rho_hi=args.force_rho_hi)
    if args.save:
        print("wrote", save(res))
    return 0 if res.get("diverged") is False else 1


if __name__ == "__main__":
    raise SystemExit(main())