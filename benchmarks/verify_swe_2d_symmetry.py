"""Symmetry / cross-solver verification of the 2-D shallow-water DG solver.

Verification-debt study of ``tensordg.swe_2d`` (TensorDG).  Four invariants that
any correct Cartesian DG discretisation of the 2-D shallow-water equations must
satisfy, none of which was asserted by an existing test:

C1  cross-solver reducibility -- a y-invariant problem solved on an ``nx x ny``
    Cartesian mesh must reproduce the 1-D solver (``tensordg.swe_1d``, which is
    independently verified against the Thacker 1981 planar-basin and Ritter dam
    break solutions) to round-off;
C2  transverse uniformity -- the m6386 defect class (a spurious y-momentum
    source in the cross mode (1,1) of a y-invariant state) must stay absent: a
    y-invariant initial state may not acquire any y-variation;
C3  index-transposition symmetry -- a radially symmetric problem solved on
    ``(nx, ny)`` and on ``(ny, nx)`` must give the same field under the
    ``(i, j, a, b) -> (j, i, b, a)`` relabelling (with ``hu <-> hv`` exchanged);
C4  well-balancedness -- still water ``h + z_b = const`` over a bed that the
    discretisation represents exactly (globally linear bed; piecewise-linear bed
    with a kink exactly on a cell face) must stay still to round-off, while the
    *smooth* bed case is recorded as a measured limitation because the bed-slope
    source uses one constant gradient per cell.

Every number in ``validation/swe_2d_symmetry/symmetry.json`` is produced here
(no hand-copied values); the two figures are written next to it.

Usage::

    PYTHONPATH=. python -m benchmarks.verify_swe_2d_symmetry               # check
    PYTHONPATH=. python -m benchmarks.verify_swe_2d_symmetry --report --figs

The script exits non-zero if any gating criterion fails.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from tensordg.swe_1d import _operators as operators_1d
from tensordg.swe_1d import solve_swe_1d
from tensordg.swe_2d import legendre_operators_2d, solve_swe_2d

OUT_DIR = os.path.join("validation", "swe_2d_symmetry")

# C1--C3 and the exactly-representable beds of C4 are *algebraic* invariants of
# the discretisation, so the only admissible deviation is floating-point
# round-off amplified over the run: 1e-11 relative is ~1e4 amplifications.
TOL_EXACT = 1.0e-11


def _recon_2d(coeff: torch.Tensor, degree: int):
    nodes, weights, phi, dphi, pl, pr = legendre_operators_2d(
        degree, coeff.device, coeff.dtype)
    return nodes, torch.einsum("xyklv,ak,bl->xyabv", coeff, phi, phi)


def _recon_1d(coeff: torch.Tensor, degree: int):
    nodes, weights, phi, dphi, pl, pr = operators_1d(
        degree, coeff.device, coeff.dtype)
    return nodes, torch.einsum("cmv,qm->cqv", coeff, phi)


def _rel(a: float, scale: float) -> float:
    return a / scale if scale else a


# ---------------------------------------------------------------------------
# C1 + C2: y-invariant dam break -- 2-D solver vs the 1-D solver
# ---------------------------------------------------------------------------
def case_cross_solver(nx: int = 16, ny: int = 4, degree: int = 1,
                      final_time: float = 0.02, cfl: float = 0.15) -> dict:
    def h0_2d(x, y):
        return torch.where(x < 0.5, torch.ones_like(x), 0.2 * torch.ones_like(x))

    def zero(x, y):
        return torch.zeros_like(x)

    t0 = time.time()
    x2, y2, c2, _zbf, meta2 = solve_swe_2d(
        h0_2d, zero, zero, zero, nx=nx, ny=ny, degree=degree,
        final_time=final_time, cfl=cfl, periodic=(False, False))
    _nodes, vals2 = _recon_2d(c2, degree)
    h2 = vals2[..., 0]
    hu2 = vals2[..., 1]

    spread_y = float((h2.max(dim=3).values - h2.min(dim=3).values).abs().max())
    spread_y_cells = float((h2.max(dim=1).values - h2.min(dim=1).values).abs().max())

    x1, c1, _zf1, meta1 = solve_swe_1d(
        lambda x: h0_2d(x, None), lambda x: zero(x, None), lambda x: zero(x, None),
        ncells=nx, degree=degree, final_time=final_time, domain=(0.0, 1.0),
        cfl=cfl, periodic=False)
    _n1, vals1 = _recon_1d(c1, degree)
    h1 = vals1[..., 0]
    hu1 = vals1[..., 1]

    dh = float((h2[:, 0, :, 0] - h1).abs().max())
    dhu = float((hu2[:, 0, :, 0] - hu1).abs().max())
    rel_h = _rel(dh, float(h1.abs().max()))
    rel_hu = _rel(dhu, float(hu1.abs().max()))
    return {
        "degree": degree, "nx": nx, "ny": ny, "final_time": final_time,
        "steps_2d": meta2["steps"], "steps_1d": meta1["steps"],
        "h_rel_diff_2d_vs_1d": rel_h, "hu_rel_diff_2d_vs_1d": rel_hu,
        "y_spread_quad": spread_y, "y_spread_cells": spread_y_cells,
        "wall_s": time.time() - t0,
        "pass_c1": rel_h <= TOL_EXACT and rel_hu <= TOL_EXACT,
        "pass_c2": max(spread_y, spread_y_cells) <= TOL_EXACT,
    }


# ---------------------------------------------------------------------------
# C3: transposition symmetry -- radial dam break on (nx, ny) vs (ny, nx)
# ---------------------------------------------------------------------------
def case_transposition(n: tuple[int, int] = (12, 20), degree: int = 1,
                       final_time: float = 0.05, cfl: float = 0.15) -> dict:
    nx, ny = n

    def h0(x, y):
        r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2
        return torch.where(r2 < 0.09, torch.ones_like(x), 0.2 * torch.ones_like(x))

    def zero(x, y):
        return torch.zeros_like(x)

    t0 = time.time()
    _x, _y, ca, _za, ma = solve_swe_2d(
        h0, zero, zero, zero, nx=nx, ny=ny, degree=degree,
        final_time=final_time, cfl=cfl)
    _x, _y, cb, _zb, mb = solve_swe_2d(
        h0, zero, zero, zero, nx=ny, ny=nx, degree=degree,
        final_time=final_time, cfl=cfl)
    _n, va = _recon_2d(ca, degree)
    _n, vb = _recon_2d(cb, degree)
    # relabelling (i, j, a, b) -> (j, i, b, a)  and  hu <-> hv
    vb_T = vb.permute(1, 0, 3, 2, 4)
    ref = va.clone()
    ref[..., 1], ref[..., 2] = va[..., 2], va[..., 1]
    diff = (ref - vb_T).abs()
    scale = float(va.abs().max())
    names = ("h", "hu", "hv")
    per_var = {names[v]: _rel(float(diff[..., v].max()), float(va[..., v].abs().max()) or scale)
               for v in range(3)}
    worst = max(per_var.values())
    return {
        "degree": degree, "cells": [nx, ny], "final_time": final_time,
        "steps": [ma["steps"], mb["steps"]], "rel_diff": per_var,
        "worst_rel_diff": worst, "wall_s": time.time() - t0,
        "pass_c3": worst <= TOL_EXACT,
    }


# ---------------------------------------------------------------------------
# C4: well-balancedness -- still water over three bed families
# ---------------------------------------------------------------------------
def _bed(kind: str, depth: float = 1.0):
    if kind == "linear":
        return lambda x, y: -0.30 * x - 0.10 * y + 0.20
    if kind == "kinked":                        # kink exactly on x = y = 0.5
        return lambda x, y: 0.20 - 0.30 * (x - 0.5).abs() - 0.10 * (y - 0.5).abs()
    if kind == "smooth":
        return lambda x, y: -0.40 * torch.exp(
            -(((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.02))
    raise ValueError(kind)


def case_well_balanced(kind: str = "linear", n: int = 16, degree: int = 1,
                       final_time: float = 0.05, cfl: float = 0.15,
                       depth: float = 1.0, gating: bool = True) -> dict:
    bed = _bed(kind, depth)

    def h_still(x, y):
        return depth - bed(x, y)

    def zero(x, y):
        return torch.zeros_like(x)

    t0 = time.time()
    _x, _y, c, _zb, meta = solve_swe_2d(
        h_still, zero, zero, bed, nx=n, ny=n, degree=degree,
        final_time=final_time, cfl=cfl)
    nodes, vals = _recon_2d(c, degree)
    centers = (torch.arange(n, dtype=c.dtype) + 0.5) / n
    X = (centers[:, None] + 0.5 * nodes[None, :] / n)[:, None, :, None]
    Y = (centers[:, None] + 0.5 * nodes[None, :] / n)[None, :, None, :]
    eta_ref = h_still(X, Y)
    err = float((vals[..., 0] - eta_ref).abs().max())
    rel = _rel(err, float(eta_ref.abs().max()))
    momentum = float(vals[..., 1:].abs().max())
    out = {
        "bed": kind, "degree": degree, "n": n, "final_time": final_time,
        "steps": meta["steps"], "max_abs_h_error": err, "rel_h_error": rel,
        "max_abs_momentum": momentum, "wall_s": time.time() - t0,
    }
    if gating:
        out["pass_c4"] = rel <= TOL_EXACT and momentum <= TOL_EXACT
    return out


def case_well_balanced_1d(kind: str = "smooth", n: int = 16, degree: int = 1,
                          final_time: float = 0.05, cfl: float = 0.15,
                          depth: float = 1.0) -> dict:
    """1-D reference for the smooth-bed limitation of the same source treatment."""
    bed2 = _bed(kind, depth)
    bed1 = lambda x: bed2(x, 0.5 * torch.ones_like(x))
    t0 = time.time()
    _x, c, _zf, meta = solve_swe_1d(
        lambda x: depth - bed1(x), lambda x: torch.zeros_like(x),
        bed1, ncells=n, degree=degree, final_time=final_time, domain=(0.0, 1.0),
        cfl=cfl, periodic=False)
    _n, vals = _recon_1d(c, degree)
    centers = (torch.arange(n, dtype=c.dtype) + 0.5) / n
    nodes = operators_1d(degree, c.device, c.dtype)[0]
    X = (centers[:, None] + 0.5 * nodes[None, :] / n)
    eta_ref = depth - bed1(X)
    err = float((vals[..., 0] - eta_ref).abs().max())
    return {
        "bed": kind + "-1d", "degree": degree, "n": n, "final_time": final_time,
        "steps": meta["steps"], "max_abs_h_error": err,
        "rel_h_error": _rel(err, float(eta_ref.abs().max())),
        "wall_s": time.time() - t0,
    }


def make_figures(res: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT_DIR, exist_ok=True)
    paths = []
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    labels, vals, keys = [], [], []
    for key, run in res["runs"].items():
        for name in ("h_rel_diff_2d_vs_1d", "y_spread_quad",
                     "worst_rel_diff", "rel_h_error"):
            if name in run:
                labels.append(f"{key}\n{name}")
                vals.append(max(run[name], 1e-18))
                keys.append(run.get("pass_c1", run.get("pass_c2",
                            run.get("pass_c3", run.get("pass_c4", None)))))
    ax[0].barh(range(len(vals)), vals)
    ax[0].set_yticks(range(len(vals)))
    ax[0].set_yticklabels(labels, fontsize=5)
    ax[0].set_xscale("log")
    ax[0].axvline(res["tol"], color="r", ls="--", lw=1, label=f"tol={res['tol']:g}")
    ax[0].set_xlabel("deviation (relative / absolute where noted)")
    ax[0].set_title("SWE-2D exact invariants vs round-off tolerance")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, which="both", axis="x")

    wb = [(k, r) for k, r in res["runs"].items() if "rel_h_error" in r]
    ax[1].bar([k.replace("well_balanced_", "") for k, _ in wb],
              [max(r["rel_h_error"], 1e-18) for _, r in wb], color="tab:blue")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("relative |h - h_exact|_inf")
    ax[1].set_title("lake at rest: exactly representable vs smooth bed")
    ax[1].axhline(res["tol"], color="r", ls="--", lw=1)
    ax[1].tick_params(axis="x", labelsize=7)
    ax[1].grid(alpha=0.3, which="both", axis="y")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "swe_2d_invariants.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description="SWE-2D symmetry verification")
    ap.add_argument("--report", action="store_true", help="write the JSON artifact")
    ap.add_argument("--figs", action="store_true", help="write the figures")
    ap.add_argument("--only", default="", help="substring filter on case keys")
    args = ap.parse_args()

    torch.set_num_threads(1)
    want = [s.strip() for s in args.only.split(",") if s.strip()]
    res: dict = {"tol": TOL_EXACT, "torch": torch.__version__,
                 "host_load": os.getloadavg()[0], "runs": {}}

    def run(key, fn):
        if want and not any(w in key for w in want):
            return
        out = fn()
        res["runs"][key] = out
        gates = [v for k, v in out.items() if k.startswith("pass_")]
        print(f"[{key}] gates={gates} " + " ".join(
            f"{k}={v:.3e}" for k, v in out.items()
            if isinstance(v, float) and k != "wall_s"), flush=True)

    t0 = time.time()
    run("cross_solver_p1", lambda: case_cross_solver(16, 4, 1))
    run("cross_solver_p2", lambda: case_cross_solver(16, 4, 2))
    run("transposition_p1", lambda: case_transposition((12, 20), 1))
    run("well_balanced_linear_p1", lambda: case_well_balanced("linear", 16, 1))
    run("well_balanced_kinked_p1", lambda: case_well_balanced("kinked", 16, 1))
    run("well_balanced_smooth", lambda: case_well_balanced("smooth", 16, 1,
                                                            gating=False))
    run("well_balanced_smooth_1d", lambda: case_well_balanced_1d("smooth", 16, 1))
    run("well_balanced_smooth_n32", lambda: case_well_balanced("smooth", 32, 1,
                                                              gating=False))
    run("well_balanced_smooth_1d_n32", lambda: case_well_balanced_1d("smooth", 32, 1))

    gates = [v for r in res["runs"].values() for k, v in r.items()
             if k.startswith("pass_")]
    res["all_pass"] = bool(gates) and all(gates)
    res["n_gates"] = len(gates)
    res["wall_s"] = time.time() - t0
    print(f"[verdict] all_pass={res['all_pass']} gates={len(gates)} "
          f"wall={res['wall_s']:.1f}s")

    if args.figs:
        try:
            print("figures:", make_figures(res))
        except Exception as exc:                     # pragma: no cover
            print("figure generation skipped:", exc)
    if args.report:
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, "symmetry.json")
        if os.path.exists(path):
            with open(path) as fh:
                prev = json.load(fh)
            prev.setdefault("runs", {}).update(res["runs"])
            prev["all_pass"] = all(v for r in prev["runs"].values()
                                   for k, v in r.items() if k.startswith("pass_"))
            prev["n_gates"] = sum(1 for r in prev["runs"].values()
                                  for k in r if k.startswith("pass_"))
            res = prev
        with open(path, "w") as fh:
            json.dump(res, fh, indent=2, sort_keys=True)
        print("artifact written:", path)
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())