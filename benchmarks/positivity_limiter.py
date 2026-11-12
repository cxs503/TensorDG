"""Verification benchmark for the Zhang-Shu positivity-preserving scaling limiter.

Case
----
Toro's test 2 ("123 problem"): a symmetric *double rarefaction* in an ideal gas,

    (rho, u, p)_L = (1, -2, 0.4),   (rho, u, p)_R = (1, +2, 0.4),  gamma = 1.4

Its star pressure is ``p* = 1.89e-3``, i.e. two orders of magnitude below the
initial pressure: the centre of the domain is a near-vacuum and a high-order
polynomial representation of the two rarefaction fans *will* go negative there
unless an admissibility treatment is applied.  Unlike a shock tube the exact
solution is analytic and smooth (two isentropic fans + a constant star region),
so the benchmark measures both **admissibility** and **accuracy retention**.

What is compared
----------------
The same modal DG/HLLC/SSP-RK3 solver (``tensordg.euler_dg_1d``) with three
admissibility treatments:

``none``      no treatment (expected to fail: non-physical state)
``collapse``  historic fallback: reset an offending cell to its average
              (first-order information loss)
``scale``     Zhang-Shu scaling limiter (new default): ``u -> ubar + theta (u - ubar)``
              with the largest admissible ``theta`` per cell

Reported per run: L1/L2 error of (rho, p, u) against the analytic solution,
minimum density/pressure over the run, centre symmetry of the velocity,
conservation defect of the cell averages, and limiter activity.

Run::

    python -m benchmarks.positivity_limiter            # full sweep
    python -m benchmarks.positivity_limiter --quick    # coarse sweep
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from tensordg.euler_dg_1d import solve_euler_dg_1d

GAMMA = 1.4
LEFT = (1.0, -2.0, 0.4)
RIGHT = (1.0, 2.0, 0.4)
DOMAIN = (0.0, 1.0)
DIAPHRAGM = 0.5
FINAL_TIME = 0.15
CFL = 0.12


# --------------------------------------------------------------------------
# Independent analytic solution (no reuse of the solver's star-state solve)
# --------------------------------------------------------------------------
def _pressure_function(p_star: float, rho_k: float, p_k: float) -> float:
    """Toro eq. (4.6): ``f(p*)`` for a rarefaction or a shock branch."""
    a_k = math.sqrt(GAMMA * p_k / rho_k)
    if p_star > p_k:  # shock
        coeff = (GAMMA + 1.0) / (2.0 * GAMMA)
        return (p_star - p_k) * math.sqrt(
            2.0 / ((GAMMA + 1.0) * rho_k) / (p_star + coeff * p_k)
        )
    # rarefaction
    return (
        2.0 * a_k / (GAMMA - 1.0)
        * ((p_star / p_k) ** ((GAMMA - 1.0) / (2.0 * GAMMA)) - 1.0)
    )


def star_pressure_bisection(left=LEFT, right=RIGHT, tol=1e-14, itmax=200) -> float:
    """Solve ``f_L(p*) + f_R(p*) = u_L - u_R`` by bisection (independent of the solver)."""
    target = left[1] - right[1]
    lo, hi = 1e-12, max(left[2], right[2])
    for _ in range(itmax):
        mid = 0.5 * (lo + hi)
        residual = (
            _pressure_function(mid, left[0], left[2])
            + _pressure_function(mid, right[0], right[2])
            - target
        )
        if residual > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def exact_double_rarefaction(x, t, p_star=None, left=LEFT, right=RIGHT, x0=DIAPHRAGM):
    """Analytic solution of the 123 problem at time ``t`` (two symmetric fans).

    Derivation from the Riemann invariants (Toro 2009, Ch. 4): inside the left
    fan ``xi = (x-x0)/t`` obeys ``u = xi + a`` and ``u + 2a/(gamma-1) =
    u_L + 2a_L/(gamma-1)``; inside the right fan ``u = xi - a`` and ``u -
    2a/(gamma-1) = u_R - 2a_R/(gamma-1)``.  Hence

        left  fan:  a = [(gamma-1)(u_L - xi) + 2 a_L] / (gamma+1),  u = xi + a
        right fan:  a = [(gamma-1)(xi - u_R) + 2 a_R] / (gamma+1),  u = xi - a

    which reproduces ``a = a_L`` at the fan head and ``a = a*`` at its tail by
    construction.  Returns a dict ``{rho, u, p, ...}``.
    """
    p_star = star_pressure_bisection(left, right) if p_star is None else p_star
    rho_l, u_l, p_l = left
    rho_r, u_r, p_r = right
    a_l = math.sqrt(GAMMA * p_l / rho_l)
    a_r = math.sqrt(GAMMA * p_r / rho_r)
    f_l = _pressure_function(p_star, rho_l, p_l)
    f_r = _pressure_function(p_star, rho_r, p_r)
    u_star = 0.5 * (u_l + u_r) + 0.5 * (f_r - f_l)
    rho_star_l = rho_l * (p_star / p_l) ** (1.0 / GAMMA)
    rho_star_r = rho_r * (p_star / p_r) ** (1.0 / GAMMA)
    a_star_l = math.sqrt(GAMMA * p_star / rho_star_l)
    a_star_r = math.sqrt(GAMMA * p_star / rho_star_r)
    head_l, tail_l = u_l - a_l, u_star - a_star_l
    tail_r, head_r = u_star + a_star_r, u_r + a_r

    x = np.asarray(x, dtype=float)
    xi = (x - x0) / t
    rho = np.empty_like(xi)
    u = np.empty_like(xi)
    p = np.empty_like(xi)

    fan_l = (xi > head_l) & (xi < tail_l)
    fan_r = (xi > tail_r) & (xi < head_r)
    star = (xi >= tail_l) & (xi <= tail_r)
    far_l = xi <= head_l
    far_r = xi >= head_r

    rho[far_l], u[far_l], p[far_l] = rho_l, u_l, p_l
    rho[far_r], u[far_r], p[far_r] = rho_r, u_r, p_r
    rho[star] = 0.5 * (rho_star_l + rho_star_r)
    u[star], p[star] = u_star, p_star

    a_left = ((GAMMA - 1.0) * (u_l - xi[fan_l]) + 2.0 * a_l) / (GAMMA + 1.0)
    u[fan_l] = xi[fan_l] + a_left
    rho[fan_l] = rho_l * (a_left / a_l) ** (2.0 / (GAMMA - 1.0))
    p[fan_l] = p_l * (a_left / a_l) ** (2.0 * GAMMA / (GAMMA - 1.0))

    a_right = ((GAMMA - 1.0) * (xi[fan_r] - u_r) + 2.0 * a_r) / (GAMMA + 1.0)
    u[fan_r] = xi[fan_r] - a_right
    rho[fan_r] = rho_r * (a_right / a_r) ** (2.0 / (GAMMA - 1.0))
    p[fan_r] = p_r * (a_right / a_r) ** (2.0 * GAMMA / (GAMMA - 1.0))

    return {"rho": rho, "u": u, "p": p, "p_star": p_star, "u_star": u_star,
            "rho_star": 0.5 * (rho_star_l + rho_star_r),
            "fans": (head_l, tail_l, tail_r, head_r)}


def fan_speeds(t=FINAL_TIME, p_star=None):
    """Wave speeds of both fans; the benchmark is valid while they stay inside."""
    sol = exact_double_rarefaction(np.array([DIAPHRAGM]), t, p_star=p_star)
    head_l, tail_l, tail_r, head_r = sol["fans"]
    x_lo = DIAPHRAGM + t * head_l
    x_hi = DIAPHRAGM + t * head_r
    return {"x_left_head": x_lo, "x_right_head": x_hi,
            "inside_domain": bool(x_lo > DOMAIN[0] and x_hi < DOMAIN[1])}


# --------------------------------------------------------------------------
# Post-processing of the modal solution
# --------------------------------------------------------------------------
def _legendre(s, order):
    """Legendre polynomials P_0..P_order at ``s`` (vectorised)."""
    s = np.asarray(s, dtype=float)
    values = [np.ones_like(s)]
    if order >= 1:
        values.append(s.copy())
    for k in range(1, order):
        values.append(((2 * k + 1) * s * values[k] - k * values[k - 1]) / (k + 1))
    return np.stack(values[: order + 1])


def evaluate(coeff: torch.Tensor, degree: int, samples: int = 40, domain=DOMAIN):
    """Evaluate the modal solution on ``samples`` points per cell -> (x, fields)."""
    ncell = coeff.shape[0]
    lower, upper = domain
    dx = (upper - lower) / ncell
    s = np.linspace(-1.0, 1.0, samples, endpoint=False) + 0.5 / samples
    basis = _legendre(s, degree)  # (m, s)
    c = coeff.detach().cpu().numpy()
    vals = np.einsum("cmv,ms->csv", c, basis)  # (cell, sample, var)
    rho = vals[..., 0]
    mom = vals[..., 1]
    energy = vals[..., 2]
    u = mom / np.where(np.abs(rho) > 1e-300, rho, 1e-300)
    p = (GAMMA - 1.0) * (energy - 0.5 * mom * u)
    x = lower + ((np.arange(ncell)[:, None] + (s[None, :] + 1.0) / 2.0) * dx)
    return x.ravel(), {"rho": rho.ravel(), "u": u.ravel(), "p": p.ravel()}


def error_norms(x, fields, t):
    exact = exact_double_rarefaction(x, t)
    out = {}
    for key in ("rho", "u", "p"):
        diff = fields[key] - exact[key]
        out[f"L1_{key}"] = float(np.mean(np.abs(diff)))
        out[f"L2_{key}"] = float(math.sqrt(np.mean(diff ** 2)))
        out[f"rel_L1_{key}"] = float(
            np.mean(np.abs(diff)) / max(np.mean(np.abs(exact[key])), 1e-300)
        )
    out["symmetry_error_u"] = float(
        np.max(np.abs(fields["u"] - fields["u"][::-1]))
    )
    out["min_rho"] = float(fields["rho"].min())
    out["min_p"] = float(fields["p"].min())
    return out


def conservation(coeff: torch.Tensor) -> dict:
    avg = coeff[:, 0].detach().cpu().numpy()
    return {name: float(avg[:, i].mean())
            for i, name in enumerate(("rho", "mom", "energy"))}


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------
def run_case(ncells, degree, positivity, final_time=FINAL_TIME):
    record = {"ncells": ncells, "degree": degree, "positivity": positivity}
    try:
        x, coeff, meta = solve_euler_dg_1d(
            LEFT, RIGHT, ncells=ncells, degree=degree, final_time=final_time,
            cfl=CFL, positivity=positivity,
        )
    except FloatingPointError as exc:  # non-physical state (expected for "none")
        record.update({"status": "NONPHYSICAL", "error": str(exc)})
        return record
    xf, fields = evaluate(coeff, degree)
    record.update(error_norms(xf, fields, final_time))
    record["conservation"] = conservation(coeff)
    record["steps"] = meta["steps"]
    record["dt"] = meta["dt"]
    record["positivity_stats"] = meta.get("positivity_stats", {})
    record["status"] = "ok"
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="coarse sweep for CI")
    # tracked evidence path: ``artifacts/`` does not survive the merge gate, so a
    # hard-asserting gate pinned there is RED on every clean checkout.
    ap.add_argument("--out", default="docs/benchmarks/results/positivity_limiter")
    ap.add_argument("--degrees", default="", help="comma separated, overrides the sweep")
    ap.add_argument("--mesh", default="", help="comma separated cell counts, overrides the sweep")
    ap.add_argument("--modes", default="none,collapse,scale")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mesh = tuple(int(v) for v in args.mesh.split(",")) if args.mesh else (
        60, 120) if args.quick else (100, 200, 400)
    degrees = tuple(int(v) for v in args.degrees.split(",")) if args.degrees else (
        (1, 2) if args.quick else (1, 2, 3))
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())

    p_star = star_pressure_bisection()
    analytic = {
        "p_star": p_star,
        "u_star": exact_double_rarefaction(np.array([DIAPHRAGM]), FINAL_TIME,
                                          p_star=p_star)["u_star"],
        "fans": exact_double_rarefaction(np.array([DIAPHRAGM]), FINAL_TIME,
                                        p_star=p_star)["fans"],
        **fan_speeds(p_star=p_star),
    }

    records = []
    for degree in degrees:
        for ncells in mesh:
            for mode in modes:
                rec = run_case(ncells, degree, mode)
                records.append(rec)
                msg = f"p={degree} N={ncells:4d} {mode:8s} "
                if rec["status"] != "ok":
                    msg += rec["status"]
                else:
                    msg += (f"L1rho={rec['L1_rho']:.3e} L1p={rec['L1_p']:.3e} "
                            f"minrho={rec['min_rho']:+.3e} minp={rec['min_p']:+.3e} "
                            f"lim={rec['positivity_stats'].get('limited_cells', 0)}")
                print(msg, flush=True)

    payload = {
        "case": {
            "name": "123 problem (Toro test 2), symmetric double rarefaction",
            "left": LEFT, "right": RIGHT, "gamma": GAMMA,
            "domain": DOMAIN, "final_time": FINAL_TIME, "cfl": CFL,
            "reference": "analytic two-rarefaction solution (Toro 2009, Ch. 4)",
        },
        "analytic": analytic,
        "records": records,
        "limiter": "Zhang & Shu, JCP 2010 / Zhang, Xia & Shu, JCP 2012 scaling limiter",
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2))

    # representative profile dump (finest mesh, highest degree, both treatments)
    finest = max(r["ncells"] for r in records)
    promode = max(degrees)
    for mode in ("collapse", "scale"):
        sel = [r for r in records if r["status"] == "ok" and r["positivity"] == mode
               and r["ncells"] == finest and r["degree"] == promode]
        if not sel:
            continue
        x, coeff, _ = solve_euler_dg_1d(LEFT, RIGHT, ncells=finest, degree=promode,
                                        final_time=FINAL_TIME, cfl=CFL, positivity=mode)
        xf, fields = evaluate(coeff, promode)
        np.savez_compressed(out / f"profile_{mode}.npz", x=xf, **fields)
    _write_figures(payload, out)
    print(f"wrote {out}/metrics.json")
    return payload


def _write_figures(payload, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib is available in CI image
        print("matplotlib unavailable: figures skipped")
        return

    records = payload["records"]
    ok = [r for r in records if r["status"] == "ok"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for mode, style in (("collapse", "o--"), ("scale", "s-")):
        for degree in sorted({r["degree"] for r in ok}):
            sel = sorted([r for r in ok if r["positivity"] == mode and r["degree"] == degree],
                         key=lambda r: r["ncells"])
            if not sel:
                continue
            axes[0].loglog([r["ncells"] for r in sel], [r["L1_rho"] for r in sel],
                           style, label=f"{mode} p={degree}")
            axes[1].semilogy([r["ncells"] for r in sel], [r["min_p"] for r in sel],
                             style, label=f"{mode} p={degree}")
    p_star = payload["analytic"]["p_star"]
    axes[1].axhline(p_star, color="k", lw=1, ls=":", label=f"p* = {p_star:.4e}")
    axes[1].set_yscale("log")
    axes[0].set_xlabel("cells"); axes[0].set_ylabel(r"$L_1$ error in $\rho$")
    axes[0].set_title("accuracy retention (123 problem)")
    axes[1].set_xlabel("cells"); axes[1].set_ylabel(r"min $p$ (evaluated)")
    axes[1].set_title("admissibility")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "accuracy_and_admissibility.png", dpi=130)

    # solution profiles for one representative run
    fig2, axes2 = plt.subplots(1, 3, figsize=(13, 3.6))
    for mode, colour in (("collapse", "tab:orange"), ("scale", "tab:blue")):
        sel = [r for r in ok if r["positivity"] == mode and r["degree"] == 2
               and r["ncells"] == max(r["ncells"] for r in ok if r["degree"] == 2)]
        if not sel:
            continue
        shipped = out / f"profile_{mode}.npz"
        if shipped.exists():
            data = np.load(shipped)
            for ax, key in zip(axes2, ("rho", "u", "p")):
                ax.plot(data["x"], data[key], color=colour, lw=1, label=mode)
    x = np.linspace(0.0, 1.0, 2001)
    exact = exact_double_rarefaction(x, FINAL_TIME)
    for ax, key in zip(axes2, ("rho", "u", "p")):
        ax.plot(x, exact[key], "k:", lw=1.2, label="exact")
        ax.set_xlabel("x"); ax.set_ylabel(key); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    axes2[0].set_title("123 problem at t=0.15 (p=2)")
    fig2.tight_layout()
    fig2.savefig(out / "profiles.png", dpi=130)


if __name__ == "__main__":
    main()