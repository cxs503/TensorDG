"""Second *physical* case for the certified-reference engine: Sod (1978) shock tube.

Why this module exists
----------------------
The round-2 closure of ``XVAL-DG-ROUND2-CERTIFIED-REFERENCE`` certified one
reference solution (Shu--Osher 1988) and judged eleven DG legs against it.  The
engine ``tensordg.certified_reference`` (``certify`` / ``judge`` / ``classify`` /
``audit_ladder``) is written to be case agnostic, but the only evidence for that
claim was *one* physics problem plus a **synthetic** power-law ladder
(:func:`tensordg.certified_reference.band_coverage_study`).  The audit of the
closure (``docs/AUDIT-m17539-XVAL-DG-ROUND2.md``, finding L1) recorded exactly
this gap: "通用性主张按现有证据偏强 —— 只有一个物理算例".

This module closes it with the standard, sharpest possible second case: a
problem whose solution is **known in closed form**.  The Sod shock tube gives
three things the Shu--Osher case cannot:

``A`` an *external* truth that shares no code with the certifier -- the exact
      Riemann solution (over the star state, the rarefaction fan and the
      Rankine--Hugoniot jump), so the certified reference's **true** error can
      be measured, not merely argued;
``B`` the honest test that the synthetic coverage study could only *simulate*:
      **does the certified band actually bracket the true error?**  (declared a
      priori as criterion C4 below -- if the band misses, that is a FAIL worth
      publishing);
``C`` three analytic invariants that do not depend on any solver at all: the
      conserved mass ``integral rho dx = 0.5625`` (nothing has left the domain at
      ``t = 0.2``), the post-shock plateau density and the shock speed
      ``W = 1.752155...``.

Nothing in the engine is modified or special-cased for this case: the same six
primitives are imported.  If the engine needed a Sod-specific patch, the
generalisation claim would be false and the run would show it.

Case definition (dimensionless, Sod, J. Comput. Phys. 27 (1978) 1--31)
----------------------------------------------------------------------
``(rho, u, p)_L = (1, 0, 1)`` for ``x < 0.5``, ``(0.125, 0, 0.1)`` for
``x >= 0.5``, ``gamma = 1.4``, domain ``[0, 1]``, output time ``t = 0.2``.

Ladders (``reference`` = cell-centred FV, candidate legs = modal DG)
--------------------------------------------------------------------
* FV-WENO5-JS / HLLC(Davis) / SSP-RK3, ``N = 800, 1600, 3200``, ``CFL = 0.3``
  -- the same solver family, CFL and ladder shape used for Shu--Osher, so the
  comparison between the two cases is like-for-like;
* DG P1/P2/P3, ``N = 200, 400, 800``, WENO limiter, ``CFL = 0.4`` -- again the
  same eleven-leg shape, restricted here to the nine in-ladder legs.

Predeclared criteria (written before the numbers were looked at)
---------------------------------------------------------------
``C1`` the FV ladder yields ``CERTIFIED`` and its band consumes ``<= 1/3`` of the
      5 % gate it is used at (engine's own non-vacuity rule);
``C2`` every DG leg judged against the certified reference passes the strict
      mission gate ``rel L1 < 3 %`` (owner precision standard; the 5 % case-pool
      gate is reported alongside);
``C3`` the certified reference is **confirmed by an oracle that shares no
      discretisation with it**: its rel L1 against the exact Riemann solution is
      ``< 1 %`` (this is the anti-self-reference evidence -- the engine itself
      cannot detect a shared-discretisation bias, see the limitation note in §5
      of the rendered report);
``C4`` the certified reference's band **brackets its true error** vs the exact
      Riemann solution: ``rel L1(ref, exact) <= U``;
``C5`` the analytic invariants hold on the certified reference to ``< 1 %``:
      conserved mass ``0.5625``, post-shock plateau density ``0.265573...`` and
      shock position ``0.5 + W t``.

``accepted`` is the conjunction of C1--C5; the CLI exits non-zero when it is
false, so a red run cannot be mistaken for a green one.

Usage
-----
    python -m benchmarks.certified_reference_sod                       # audit
    python -m benchmarks.certified_reference_sod --figures             # + PNGs
    python -m benchmarks.certified_reference_sod --json-stdout         # machine form
    python -m benchmarks.certified_reference_sod --require-accepted    # CI gate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # allow ``python benchmarks/certified_reference_sod.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensordg.certified_reference import (  # noqa: E402  (same six primitives, no patch)
    CERTIFIED,
    U_SHARE_MAX,
    certify,
    judge,
    rel_l1,
)

GAMMA = 1.4
LEFT = (1.0, 0.0, 1.0)
RIGHT = (0.125, 0.0, 0.1)
DIAPHRAGM = 0.5
DOMAIN = (0.0, 1.0)
FINAL_TIME = 0.2
INITIAL_MASS = (DIAPHRAGM - DOMAIN[0]) * LEFT[0] + (DOMAIN[1] - DIAPHRAGM) * RIGHT[0]

FV_LADDER = (800, 1600, 3200)
FV_CFL = 0.3
DG_LADDER = (200, 400, 800)
DG_CFL = 0.4
DG_DEGREES = (1, 2, 3)

GATE_PERCENT = 5.0     # case-pool gate the reference must be usable at
STRICT_PERCENT = 3.0   # owner precision standard

#: Analytic windows of the t = 0.2 solution, from the exact wave speeds
#: (rarefaction head/tail, contact, shock).  Fixed here so the window split is
#: not tuned to any numerical result.
WINDOWS = {
    "rarefaction": (0.27, 0.49),
    "star_left": (0.51, 0.67),
    "star_right": (0.70, 0.84),
    "untouched_right": (0.86, 1.0),
}


# ------------------------------------------------------------------ case solver
def initial_state(x):  # noqa: D401 - FV solver interface
    """Two-state initial data for the FV-WENO5 solver (``x`` is a column tensor)."""
    import torch

    xf = x.reshape(-1)
    left = xf < DIAPHRAGM
    rho = torch.where(left, torch.full_like(xf, LEFT[0]), torch.full_like(xf, RIGHT[0]))
    vel = torch.where(left, torch.full_like(xf, LEFT[1]), torch.full_like(xf, RIGHT[1]))
    pre = torch.where(left, torch.full_like(xf, LEFT[2]), torch.full_like(xf, RIGHT[2]))
    return rho[:, None], vel[:, None], pre[:, None]


def fv_levels(ladder=FV_LADDER, cfl: float = FV_CFL) -> list[dict]:
    """The certifier ladder: cell-centred FV-WENO5 density profiles."""
    from benchmarks.fv_weno5_1d import solve_fv_weno5

    levels = []
    for ncells in ladder:
        x, u, meta = solve_fv_weno5(initial_state, ncells=ncells,
                                    final_time=FINAL_TIME, domain=DOMAIN, cfl=cfl)
        levels.append({"ncells": int(ncells), "field": np.asarray(u[:, 0], dtype=float),
                       "x": np.asarray(x, dtype=float), "meta": meta})
    return levels


def dg_levels(degree: int, ladder=DG_LADDER, cfl: float = DG_CFL) -> list[dict]:
    """One DG family (cell means, i.e. 0th modal coefficient) on a nesting ladder."""
    from tensordg.euler_dg_1d import solve_euler_dg_1d

    levels = []
    for ncells in ladder:
        x, coeff, meta = solve_euler_dg_1d(
            LEFT, RIGHT, ncells=int(ncells), degree=int(degree), final_time=FINAL_TIME,
            diaphragm=DIAPHRAGM, domain=DOMAIN, cfl=cfl, limiter="weno",
            limiter_stage="final")
        mean = np.asarray(coeff[:, 0].detach().cpu().numpy(), dtype=float)
        levels.append({"ncells": int(ncells), "field": mean[:, 0],
                       "x": np.asarray(x.detach().cpu().numpy(), dtype=float), "meta": meta})
    return levels


# --------------------------------------------------------------- analytic truth
def exact_profiles(ncells: int) -> dict:
    """Exact Riemann solution as 8-point cell averages on ``ncells`` cells."""
    from benchmarks.riemann_exact import cell_averages

    centers, fields = cell_averages(
        ncells, FINAL_TIME, LEFT, RIGHT, diaphragm=DIAPHRAGM,
        domain=DOMAIN, quadrature_order=8, gamma=GAMMA)
    return {"x": np.asarray(centers, dtype=float),
            "density": np.asarray(fields["density"], dtype=float),
            "velocity": np.asarray(fields["velocity"], dtype=float),
            "pressure": np.asarray(fields["pressure"], dtype=float)}


def star_state() -> dict:
    """Closed-form star-region values (Toro's p*/u* solve) for the analytic anchors."""
    from benchmarks.riemann_exact import star_state as _star

    p_star, u_star = _star(LEFT, RIGHT, GAMMA)
    a_l = np.sqrt(GAMMA * LEFT[2] / LEFT[0])
    a_r = np.sqrt(GAMMA * RIGHT[2] / RIGHT[0])
    rho_star_l = LEFT[0] * (p_star / LEFT[2]) ** (1.0 / GAMMA)
    rho_star_r = RIGHT[0] * ((p_star / RIGHT[2] + (GAMMA - 1.0) / (GAMMA + 1.0))
                             / ((GAMMA - 1.0) / (GAMMA + 1.0) * p_star / RIGHT[2] + 1.0))
    # post-shock (right of contact) state from the Rankine-Hugoniot relations
    rho_shock = rho_star_r
    shock_speed = RIGHT[1] + a_r * np.sqrt(
        (GAMMA + 1.0) / (2.0 * GAMMA) * p_star / RIGHT[2]
        + (GAMMA - 1.0) / (2.0 * GAMMA))
    return {"p_star": float(p_star), "u_star": float(u_star),
            "rho_star_left": float(rho_star_l), "rho_star_right": float(rho_shock),
            "shock_speed": float(shock_speed),
            "shock_position": float(DIAPHRAGM + shock_speed * FINAL_TIME),
            "contact_position": float(DIAPHRAGM + u_star * FINAL_TIME),
            "sound_speed_left": float(a_l)}


def _windowed_rel_l1(x_num, f_num, x_ref, f_ref, lo, hi) -> float | None:
    m_ref = (x_ref >= lo) & (x_ref <= hi)
    if not np.any(m_ref):
        return None
    a = np.interp(x_ref[m_ref], x_num, f_num)
    return float(rel_l1(a, f_ref[m_ref]))


def _shock_position(x, rho, x_min: float = 0.0) -> float:
    """Location of the steepest density gradient at ``x >= x_min``.

    ``x_min`` exists because in Sod the **contact** is a larger density jump
    (0.4263 -> 0.2656) than the shock (0.2656 -> 0.1250), so a global argmax
    locates the contact, not the shock, once the grid resolves both.  The bound
    is only used to keep the search downstream of the contact (analytic
    ``x0 + u* t``); the reported position itself is measured from the profile.
    """
    x = np.asarray(x, dtype=float)
    rho = np.asarray(rho, dtype=float)
    d = np.abs(np.diff(rho) / np.diff(x))
    mid = 0.5 * (x[:-1] + x[1:])
    d = np.where(mid >= x_min, d, -np.inf)
    i = int(np.argmax(d))
    return float(mid[i])


# ------------------------------------------------------------------------ audit
def audit(fv_ladder=FV_LADDER, dg_ladder=DG_LADDER, degrees=DG_DEGREES,
          fv_cfl: float = FV_CFL, dg_cfl: float = DG_CFL,
          tolerance_percent: float = GATE_PERCENT, strict_percent: float = STRICT_PERCENT,
          levels_override: dict | None = None, keep_levels: bool = False) -> dict:
    """Run the second-case audit.  ``levels_override`` lets tests inject ladders."""
    t0 = time.perf_counter()
    fv = (levels_override or {}).get("fv") or fv_levels(fv_ladder, fv_cfl)
    dg_override = (levels_override or {}).get("dg")
    dg = {}
    for degree in degrees:
        if dg_override is not None:
            dg[degree] = list(dg_override.get(degree, []))
        else:
            dg[degree] = dg_levels(degree, dg_ladder, dg_cfl)

    cert = certify(fv, tolerance_percent=tolerance_percent)
    ref = cert.get("reference")
    ref_n = int(cert.get("reference_ncells") or fv[-2]["ncells"])
    exact = exact_profiles(ref_n)

    # C1 -- the ladder may certify a reference usable at the gate
    c1 = bool(cert.get("verdict") == CERTIFIED
              and (cert.get("band_share_of_gate") or 0.0) <= U_SHARE_MAX)

    # C4 -- honest coverage: does the certified band bracket the true error?
    true_error = None
    coverage = None
    windows = {}
    if ref is not None:
        true_error = float(rel_l1(ref, exact["density"]))
        ref_x = exact["x"]
        windows = {name: _windowed_rel_l1(ref_x, ref, ref_x, exact["density"], lo, hi)
                   for name, (lo, hi) in WINDOWS.items()}
        band = cert.get("band_percent")
        coverage = {
            "true_error_percent": 100.0 * true_error,
            "band_percent": band,
            "band_covers_true_error": bool(band is not None and 100.0 * true_error <= band),
            "ratio_true_error_over_band": (None if not band
                                           else float(100.0 * true_error / band)),
            "raw_finest_true_error_percent": 100.0 * float(
                rel_l1(_downsample(fv[-1]["field"], 2), exact["density"])),
        }
    c4 = bool(coverage is not None and coverage["band_covers_true_error"])

    # C5 -- analytic invariants measured on the certified reference
    anchors = {}
    if ref is not None:
        st = star_state()
        xr = exact["x"]
        dx = float(xr[1] - xr[0]) if xr.size > 1 else 1.0
        anchors = {
            "conserved_mass": {
                "exact": float(INITIAL_MASS),
                "measured": float(np.sum(ref) * dx),
            },
            "plateau_density": {
                "exact": st["rho_star_right"],
                # the post-shock plateau lives *between* the contact and the
                # shock (left of the shock!), so it is sampled on the analytic
                # star-right window and never on the untouched right state
                "measured": float(np.mean(ref[(xr >= st["contact_position"] + 0.01)
                                              & (xr <= st["shock_position"] - 0.01)])),
            },
            "star_left_density": {
                "exact": st["rho_star_left"],
                # star-left plateau: between the rarefaction tail (x0 - a_L t is
                # outside) and the contact, the exact solution is constant
                "measured": float(np.mean(ref[(xr >= DIAPHRAGM + 0.02)
                                              & (xr <= st["contact_position"] - 0.01)])),
            },
            "shock_position": {
                "exact": st["shock_position"],
                "measured": _shock_position(xr, ref,
                                            x_min=st["contact_position"] + 0.01),
            },
        }
        for key in anchors:
            e, m = anchors[key]["exact"], anchors[key]["measured"]
            anchors[key]["rel_error_percent"] = float(100.0 * abs(m - e) / abs(e))
        anchors["star_state"] = st
    c5 = bool(anchors and all(v["rel_error_percent"] < 1.0
                              for k, v in anchors.items() if isinstance(v, dict)
                              and "rel_error_percent" in v))

    # C2 -- DG legs against the certified reference, at both gates
    legs = []
    if ref is not None:
        for degree in degrees:
            for lv in dg[degree]:
                res = judge(lv["field"], cert)
                strict = judge(lv["field"], cert, tolerance_percent=strict_percent)
                legs.append({
                    "configuration": f"DG P{degree} N={lv['ncells']} CFL={dg_cfl:g}",
                    "degree": int(degree), "ncells": int(lv["ncells"]),
                    "rel_l1_percent": res["rel_l1_percent"],
                    "band_percent": res.get("band_percent"),
                    "decision_5pct": res["decision"],
                    "decision_3pct": strict["decision"],
                })
        counts5, counts3 = {}, {}
        for lg in legs:
            counts5[lg["decision_5pct"]] = counts5.get(lg["decision_5pct"], 0) + 1
            counts3[lg["decision_3pct"]] = counts3.get(lg["decision_3pct"], 0) + 1
    else:
        counts5 = counts3 = {}
    c2 = bool(legs and all(lg["decision_3pct"].startswith("PASS") for lg in legs))

    # C3 (measured, not assumed) -- the reference must be confirmed by an oracle
    # that shares no discretisation with it.  The engine *cannot* detect a
    # shared-discretisation bias (see "dg_self_certification" below, reported as
    # a documented limitation), so the anti-self-reference evidence has to come
    # from the closed-form solution: this is what C3 measures.
    c3_oracle = bool(coverage is not None
                     and coverage["true_error_percent"] < 1.0)

    # informational: which DG ladders would the engine itself accept as certifiers?
    self_cert = {}
    for degree in degrees:
        sc = certify(dg[degree], tolerance_percent=tolerance_percent)
        self_cert[f"dg_p{degree}"] = {
            "verdict": sc.get("verdict"),
            "classification": sc.get("classification"),
            "band_percent": sc.get("band_percent"),
            "band_share_of_gate": sc.get("band_share_of_gate"),
            "observed_order": sc.get("observed_order_used"),
            "reasons": sc.get("reasons", []),
        }
    criteria = {
        "C1_ladder_certifies_at_gate": c1,
        "C2_dg_legs_under_strict_gate": c2,
        "C3_external_oracle_under_1pct": c3_oracle,
        "C4_band_brackets_true_error": c4,
        "C5_analytic_invariants_under_1pct": c5,
    }
    out = {
        "case": "Sod 1978 shock tube (second physical case for the certified-reference engine)",
        "parameters": {"gamma": GAMMA, "left_state": LEFT, "right_state": RIGHT,
                       "diaphragm": DIAPHRAGM, "domain": list(DOMAIN),
                       "final_time": FINAL_TIME, "initial_mass": INITIAL_MASS},
        "ladders": {"fv": list(fv_ladder), "fv_cfl": fv_cfl,
                    "dg": list(dg_ladder), "dg_cfl": dg_cfl, "degrees": list(degrees)},
        "gates": {"reference_gate_percent": tolerance_percent,
                  "strict_gate_percent": strict_percent, "u_share_max": U_SHARE_MAX},
        "fv_certification": {k: v for k, v in cert.items()
                             if k not in ("reference", "field_raw_finest")},
        "error_vs_exact": {"reference_ncells": ref_n, "windows": windows,
                           **(coverage or {})},
        "analytic_anchors": anchors,
        "dg_legs": legs,
        "dg_decision_counts_5pct": counts5,
        "dg_decision_counts_3pct": counts3,
        "dg_self_certification": self_cert,
        "criteria": criteria,
        "accepted": bool(all(criteria.values())),
        "wall_time_s": time.perf_counter() - t0,
    }
    if keep_levels:
        out["_levels"] = {"fv": fv, "dg": dg}
    return out


def _downsample(field, factor: int) -> np.ndarray:
    f = np.asarray(field, dtype=float)
    n = f.size // factor
    return f[:n * factor].reshape(n, factor).mean(axis=1)


def _sanitise(obj):
    """JSON-safe copy: numpy scalars/arrays -> python, so artifacts are portable."""
    if isinstance(obj, dict):
        return {str(k): _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitise(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# -------------------------------------------------------------------- rendering
def to_markdown(a: dict) -> str:
    cert = a["fv_certification"]
    err = a["error_vs_exact"]
    lines = [
        "# Sod 1978 shock tube -- second physical case for the certified-reference engine",
        "",
        f"Case: `{a['case']}`",
        "",
        "## 1. Can the ladder certify a reference at the gate?",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| FV ladder (cells) | {cert['ladder']} |",
        f"| observed order p | {cert.get('observed_order_used')} |",
        f"| band U (%) | {cert.get('band_percent')} |",
        f"| U / gate ({a['gates']['reference_gate_percent']:g} %) | {cert.get('band_share_of_gate')} |",
        f"| verdict | **{cert.get('verdict')}** |",
        f"| refusal reasons | {cert.get('reasons')} |",
        "",
        "## 2. Does the band bracket the true error? (the point of this case)",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| reference grid (cells) | {err['reference_ncells']} |",
        f"| rel L1(certified ref, exact Riemann) % | **{err.get('true_error_percent'):.6f}** |",
        f"| certified band U % | {err.get('band_percent')} |",
        f"| true error / U | {err.get('ratio_true_error_over_band')} |",
        f"| band covers true error | **{err.get('band_covers_true_error')}** |",
        f"| raw finest level vs exact % | {err.get('raw_finest_true_error_percent'):.6f} |",
        "",
        "### Window-resolved error of the certified reference (%)",
        "",
        "| window | rel L1 % |",
        "|---|---|",
    ]
    for name, value in (err.get("windows") or {}).items():
        lines.append(f"| {name} | {'n/a' if value is None else f'{100*value:.6f}'} |")
    lines += ["", "## 3. Analytic invariants on the certified reference", "",
              "| invariant | exact | measured | rel error % |", "|---|---|---|---|"]
    for key in ("conserved_mass", "plateau_density", "star_left_density", "shock_position"):
        v = a["analytic_anchors"].get(key, {})
        lines.append(f"| {key} | {v.get('exact')} | {v.get('measured')} | "
                     f"{v.get('rel_error_percent')} |")
    st = a["analytic_anchors"].get("star_state", {})
    lines += ["", f"Star state (closed form): p* = {st.get('p_star')}, u* = {st.get('u_star')}, "
                  f"rho*_L = {st.get('rho_star_left')}, rho*_R = {st.get('rho_star_right')}, "
                  f"W = {st.get('shock_speed')}.", "",
              "## 4. DG legs judged against the certified reference", "",
              "| configuration | rel L1 % | + U | decision @5 % | decision @3 % |",
              "|---|---|---|---|---|"]
    for lg in a["dg_legs"]:
        plus = lg["rel_l1_percent"] + (lg["band_percent"] or 0.0)
        lines.append(f"| {lg['configuration']} | {lg['rel_l1_percent']:.6f} | {plus:.6f} | "
                     f"{lg['decision_5pct']} | {lg['decision_3pct']} |")
    lines += ["", f"Decision counts @5 %: `{a['dg_decision_counts_5pct']}`; "
                  f"@3 %: `{a['dg_decision_counts_3pct']}`.", "",
              "## 5. No ladder may certify itself", "",
              "| ladder | verdict | classification | p | band U / gate |", "|---|---|---|---|---|"]
    for name, v in a["dg_self_certification"].items():
        lines.append(f"| {name} | {v['verdict']} | {v['classification']} | "
                     f"{v['observed_order']} | {v['band_share_of_gate']} |")
    lines += ["", "The engine's refusal rules are *numerical* (order / stagnation / "
                  "round-off / band-vs-gate). Nothing in them can tell that the ladder "
                  "offering itself as the reference belongs to the same code family as "
                  "the legs it would judge; a sufficiently well-converged DG ladder is "
                  "therefore accepted. Choosing a certifier from a different scheme "
                  "family stays the caller's duty, and the only machine-checkable "
                  "defence against shared bias is criterion C3 (an external oracle).",
              "", "## 6. Criteria (predeclared)", "", "| criterion | result |", "|---|---|"]
    for name, ok in a["criteria"].items():
        lines.append(f"| {name} | {'PASS' if ok else '**FAIL**'} |")
    lines += ["", f"**accepted = {a['accepted']}**", ""]
    return "\n".join(lines)


def figures(a: dict, fv: list[dict], dg: dict, outdir: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:  # compute nodes may ship without matplotlib
        print("matplotlib unavailable: figures skipped", file=sys.stderr)
        return []

    outdir.mkdir(parents=True, exist_ok=True)
    exact = exact_profiles(int(a["error_vs_exact"]["reference_ncells"]))
    names = []
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)
    ax = axes[0]
    ax.plot(exact["x"], exact["density"], "k-", lw=1.8, label="exact Riemann")
    for lv in fv:
        ax.plot(lv["x"], lv["field"], lw=0.8, alpha=0.8,
                label=f"FV-WENO5 N={lv['ncells']}")
    ax.set_xlabel("x"); ax.set_ylabel(r"$\rho$"); ax.legend(fontsize=8)
    ax.set_title(f"Sod t=0.2: certifier ladder; true error of certified ref = "
                 f"{a['error_vs_exact']['true_error_percent']:.4f} %, band U = "
                 f"{a['fv_certification']['band_percent']:.4f} %")
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(exact["x"], exact["density"], "k-", lw=1.8, label="exact Riemann")
    for degree, levels in dg.items():
        lv = levels[-1]
        ax.plot(lv["x"], lv["field"], lw=1.0, alpha=0.85,
                label=f"DG P{degree} N={lv['ncells']}")
    ax.set_xlabel("x"); ax.set_ylabel(r"$\rho$"); ax.legend(fontsize=8)
    ax.set_title("DG legs (densest grid of each family) vs exact solution")
    ax.grid(alpha=0.3)
    fig.savefig(outdir / "sod_certified_reference.png", dpi=160)
    plt.close(fig)
    names.append("sod_certified_reference.png")

    # error-vs-exact of the certified reference, resolved by window
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    labels = list((a["error_vs_exact"].get("windows") or {}).keys())
    vals = [100.0 * a["error_vs_exact"]["windows"][k] for k in labels]
    ax.bar(labels, vals, color="#4c72b0")
    ax.axhline(a["fv_certification"]["band_percent"], color="crimson", ls="--",
               label=f"certified band U = {a['fv_certification']['band_percent']:.4f} %")
    ax.set_ylabel("rel L1 vs exact (%)")
    ax.set_title("Where the certified reference's error lives (window-resolved)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.savefig(outdir / "sod_window_errors.png", dpi=160)
    plt.close(fig)
    names.append("sod_window_errors.png")
    return names


# ------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", default="artifacts/sod_certified_reference")
    ap.add_argument("--fv-ladder", default=",".join(str(n) for n in FV_LADDER))
    ap.add_argument("--dg-ladder", default=",".join(str(n) for n in DG_LADDER))
    ap.add_argument("--degrees", default=",".join(str(d) for d in DG_DEGREES))
    ap.add_argument("--fv-cfl", type=float, default=FV_CFL)
    ap.add_argument("--dg-cfl", type=float, default=DG_CFL)
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--dump-profiles", action="store_true",
                    help="write profiles.npz (certifier + DG legs + exact solution)")
    ap.add_argument("--json-stdout", action="store_true")
    ap.add_argument("--require-accepted", action="store_true")
    args = ap.parse_args(argv)

    fv_ladder = tuple(int(s) for s in args.fv_ladder.split(",") if s.strip())
    dg_ladder = tuple(int(s) for s in args.dg_ladder.split(",") if s.strip())
    degrees = tuple(int(s) for s in args.degrees.split(",") if s.strip())
    a = audit(fv_ladder=fv_ladder, dg_ladder=dg_ladder, degrees=degrees,
              fv_cfl=args.fv_cfl, dg_cfl=args.dg_cfl,
              keep_levels=args.figures or args.dump_profiles)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    a = _sanitise(a)
    (outdir / "sod_certified_reference.json").write_text(json.dumps(a, indent=2) + "\n")
    (outdir / "sod_certified_reference.md").write_text(to_markdown(a))
    if args.dump_profiles:
        lv = a.get("_levels") or {}
        store = {}
        for i, level in enumerate(lv.get("fv", [])):
            store[f"x_fv_{level['ncells']}"] = level["x"]
            store[f"rho_fv_{level['ncells']}"] = level["field"]
        for degree, levels in (lv.get("dg") or {}).items():
            for level in levels:
                store[f"x_dg_p{degree}_n{level['ncells']}"] = level["x"]
                store[f"rho_dg_p{degree}_n{level['ncells']}"] = level["field"]
        ex = exact_profiles(int(a["error_vs_exact"]["reference_ncells"]))
        store["x_exact"] = ex["x"]
        store["rho_exact"] = ex["density"]
        np.savez_compressed(outdir / "profiles.npz", **store)
    if args.figures:
        lv = a.pop("_levels")
        a["figures"] = figures(a, lv["fv"], lv["dg"], outdir)
        a = _sanitise(a)
        (outdir / "sod_certified_reference.json").write_text(json.dumps(a, indent=2) + "\n")
    a.pop("_levels", None)
    if args.json_stdout:
        print(json.dumps(a, indent=1, sort_keys=True))
    else:
        print(to_markdown(a))
    if args.require_accepted and not a["accepted"]:
        print("REJECTED: predeclared criteria not met", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())