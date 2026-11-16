"""Band honesty of :mod:`tensordg.certified_reference` in the *shock* regime.

Why this file exists
--------------------
``band_coverage_study`` (in :mod:`tensordg.certified_reference`) answers "does the
GCI band ``U`` really cover the certified profile's own error?", but it says so in
its own docstring that it **cannot answer it where it matters**::

    Limitation: the smooth ladders here recover a high observed order, so the
    extrapolation removes almost all of the error and the band is *very*
    conservative.  That says nothing about the shock case, where the true error
    cannot be measured at all (no exact solution) and the observed order is ~1.

That is exactly the regime the shipped round-2 reference lives in: CASE-SHU-OSHER
is certified at ``p = 1.1767`` (shock/entropy-wave interaction), and every DG leg
of the case is judged against that reference at a 3 % gate.  A band honesty claim
that holds only for smooth ladders therefore does not cover the case it is used
for.

What is measured here (and why it is falsifiable)
-------------------------------------------------
A **manufactured** truth is used, so the certified profile's error is *known*
rather than assumed:

* truth ``f(x) = s0 + s1 sin(k x) + J H(x - x_s)`` -- one discontinuity plus a
  smooth part; its **exact cell averages** on any grid are available in closed
  form (``_cell_averages``), so "the answer" is a grid-independent object and
  comparing two grids never introduces reconstruction ambiguity;
* a modelled numerical solution that *is* what a shock-capturing scheme returns:
  the jump is smeared over ``w = beta * h`` (``beta = 2`` is a first-order
  upwind/Rusanov-like profile, ``beta ~ 1`` a sharper one).  The error field is
  localised in O(1) cells and its L1 size is O(h), i.e. observed order ~1 -- the
  Shu-Osher regime;
* the certified profile is built by the **shipped engine** (``certify``), not by
  a reimplementation, and its error is measured against the exact cell averages
  of the truth **on the reference grid**.

Three machine questions are answered per case:

1. ``certified``   -- does the engine even certify a shock ladder (verdict), and
   at what observed order?
2. ``coverage``    -- is the true error of the certified profile <= its own band
   ``U``?  This is the same falsifiable statement ``band_coverage_study`` makes,
   now in the regime that matters.
3. ``false_pass``  -- feed a *deliberately wrong* candidate (a smeared jump of the
   wrong width, i.e. a solution that is genuinely worse than the gate) to
   ``judge``.  Does the shipped decision produce ``PASS`` / ``PASS_MARGINAL``
   even though the candidate's true error exceeds the gate?  The vacuity rule
   (``U / tau <= 1/3``) is supposed to make that impossible; here it is measured
   instead of asserted.

Controls
--------
* ``smooth`` regime: single-term smooth error model ``A g(x) N^-p`` with known
  ``p`` -- reproduces the existing study's favourable result, so a difference in
  the shock regime cannot be blamed on the harness;
* ``exact`` regime: the truth's own cell averages as all levels (zero error) --
  the engine must *refuse* it (``WITHHELD_ROUNDOFF_FLOOR``), not certify a
  number with no order information.

Usage::

    python -m benchmarks.certified_reference_shock_coverage --outdir artifacts/x

Outputs ``shock_coverage.json`` + ``shock_coverage.md`` (numbers injected
programmatically, never hand-copied) and, with ``--figures``, PNG diagnostics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tensordg.certified_reference import (
    CERTIFIED,
    WITHHELD_ROUNDOFF_FLOOR,
    certify,
    judge,
    rel_l1,
)

#: Largest ``true_error / band`` ratio measured over the certified shock ladders of
#: this study (see ``docs/XVAL-DG-ROUND2-SHOCK-REGIME-BAND-HONESTY.md``).  A
#: shock-regime reference may therefore claim ``err < k * U`` but **not**
#: ``err < U``; the smooth regime over-covers by ~3 orders of magnitude, which is
#: why the smooth-only coverage study could not have found this.
BAND_FACTOR_SHOCK = 4.0

__all__ = [
    "BAND_FACTOR_SHOCK",
    "smooth_integral",
    "ramp_integral",
    "jump_integral",
    "cell_averages",
    "modeled_levels",
    "study_case",
    "run_study",
    "write_markdown",
]


# --------------------------------------------------------------- exact integrals
def smooth_integral(a, b, s0: float, s1: float, k: float):
    """``int_a^b s0 + s1 sin(k x) dx`` (closed form, vectorised)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return s0 * (b - a) + (s1 / k) * (np.cos(k * a) - np.cos(k * b))


def _ramp_antiderivative(x, xs: float, w: float):
    """``A(x) = int_{-inf}^{x} clamp((t - xs)/w + 1/2, 0, 1) dt``.

    ``A`` is ``0`` left of the ramp, quadratic inside it and linear (slope 1) to
    the right, which is what makes the *cell average* of a smeared jump exact
    with no quadrature error at all -- the study must not confuse its own
    quadrature with the solver's error.
    """
    x = np.asarray(x, dtype=float)
    left = xs - 0.5 * w
    right = xs + 0.5 * w
    out = np.zeros_like(x)
    inside = (x > left) & (x < right)
    out[inside] = (x[inside] - left) ** 2 / (2.0 * w)
    beyond = x >= right
    out[beyond] = 0.5 * w + (x[beyond] - right)
    return out


def ramp_integral(a, b, xs: float, w: float):
    """``int_a^b clamp((x - xs)/w + 1/2, 0, 1) dx`` (exact)."""
    if w <= 0.0:
        return jump_integral(a, b, xs)
    return _ramp_antiderivative(b, xs, w) - _ramp_antiderivative(a, xs, w)


def jump_integral(a, b, xs: float):
    """``int_a^b H(x - xs) dx`` (exact)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.maximum(0.0, b - np.maximum(a, xs))


def cell_averages(ncells: int, *, xs: float = 0.41, jump: float = 0.0,
                  width: float = 0.0, s0: float = 1.0, s1: float = 0.2,
                  k: float = 6.0, length: float = 1.0) -> np.ndarray:
    """Exact cell averages of ``s0 + s1 sin(kx) + jump * ramp((x-xs)/width)``.

    ``width == 0`` gives the true discontinuous field (``H(x - xs)``); a finite
    ``width`` gives the smeared field a shock-capturing scheme returns.  Because
    the antiderivatives are exact, ``coarsen`` of these vectors equals the
    averages on the coarser grid to machine precision -- nesting carries no
    ambiguity, so every difference reported below is solver error, not
    interpolation.
    """
    n = int(ncells)
    h = length / n
    edges = np.linspace(0.0, length, n + 1)
    a, b = edges[:-1], edges[1:]
    integral = smooth_integral(a, b, s0, s1, k)
    if jump:
        integral = integral + jump * (
            jump_integral(a, b, xs) if width <= 0.0 else ramp_integral(a, b, xs, width))
    return integral / h


def modeled_levels(ncells, *, beta: float = 2.0, **kw):
    """Shock-capturing ladder: jump smeared over ``beta * h`` at every level."""
    return [{"ncells": int(n),
             "field": cell_averages(int(n), width=beta / float(n), **kw)}
            for n in ncells]


def shifted_levels(ncells, *, shift: float = 0.5, **kw):
    """Second shock model: the captured jump sits ``shift * h`` off the truth.

    A capturing scheme does not only smear a jump, it also places it within a
    cell (upwind-biased).  The error field is then a localized dipole of width
    ``h`` and amplitude ``J``, converging in L1 at ``O(h)`` -- the same order as
    the smearing model but with a completely different spatial structure, so an
    under-coverage result common to both is not an artefact of one model.
    """
    xs = float(kw.pop("xs"))
    return [{"ncells": int(n),
             "field": cell_averages(int(n), xs=xs + shift / float(n), **kw)}
            for n in ncells]


def smooth_error_levels(ncells, *, p: float = 2.0, amplitude: float = 0.05, **kw):
    """Smooth control: exact averages + ``A g(x) N^-p`` (single-term model)."""
    out = []
    for n in ncells:
        n = int(n)
        x = (np.arange(n) + 0.5) / n
        g = np.sin(3.0 * np.pi * x)
        out.append({"ncells": n,
                    "field": cell_averages(n, **kw) + amplitude * g * n ** (-p)})
    return out


def exact_levels(ncells, **kw):
    """Zero-error control: the truth's own cell averages at every level."""
    return [{"ncells": int(n), "field": cell_averages(int(n), **kw)} for n in ncells]


# --------------------------------------------------------------------- one case
def _true_error_percent(profile, ncells: int, truth_kw: dict) -> float:
    """Relative L1 of ``profile`` against the exact cell averages of the truth."""
    return 100.0 * rel_l1(np.asarray(profile, dtype=float),
                          cell_averages(int(ncells), **truth_kw))


def study_case(*, regime: str, ladder=(50, 100, 200), tolerance_percent: float = 3.0,
               beta: float = 2.0, xs: float = 0.41, jump: float = 0.0,
               s1: float = 0.2, k: float = 6.0, error_order: float = 2.0) -> dict:
    """Certify one manufactured ladder and audit its band + the judge decision."""
    truth_kw = {"xs": xs, "jump": jump, "width": 0.0, "s1": s1, "k": k}
    if regime == "shock":
        levels = modeled_levels(ladder, beta=beta, xs=xs, jump=jump, s1=s1, k=k)
    elif regime == "shift":
        levels = shifted_levels(ladder, shift=beta, xs=xs, jump=jump, s1=s1, k=k)
    elif regime == "smooth":
        levels = smooth_error_levels(ladder, p=error_order, xs=xs, jump=0.0,
                                     s1=s1, k=k)
    elif regime == "exact":
        levels = exact_levels(ladder, xs=xs, jump=jump, s1=s1, k=k)
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unknown regime {regime!r}")

    cert = certify(levels, tolerance_percent=tolerance_percent)
    d_fine = None
    if cert["differences"]:
        d_fine = cert["differences"][-1]["rel_l1_percent"]
    # the practice the engine exists to replace: quoting the finest grid itself
    # as "the reference".  Its true error is not visible from the ladder.
    raw_finest_true = _true_error_percent(levels[-1]["field"], int(ladder[-1]), truth_kw)
    row = {
        "regime": regime,
        "beta": float(beta),
        "jump": float(jump),
        "xs": float(xs),
        "s1": float(s1),
        "k": float(k),
        "ladder": [int(n) for n in ladder],
        "tolerance_percent": float(tolerance_percent),
        "verdict": cert["verdict"],
        "classification": cert["classification"],
        "certified": bool(cert["verdict"] == CERTIFIED),
        "observed_order": cert["observed_order_used"],
        "band_percent": cert["band_percent"],
        "band_share_of_gate": cert["band_share_of_gate"],
        "refinement_fine": cert["refinement_fine"],
        "d_fine_percent": d_fine,
        "raw_finest_true_error_percent": raw_finest_true,
        "masked_error_factor": (raw_finest_true / d_fine) if d_fine else None,
        "true_error_percent": None,
        "coverage": None,
        "true_over_band": None,
    }
    if cert["reference"] is not None:
        te = _true_error_percent(cert["reference"], cert["reference_ncells"], truth_kw)
        row["true_error_percent"] = te
        if cert["band_percent"]:
            row["coverage"] = bool(te <= cert["band_percent"])
            row["true_over_band"] = te / cert["band_percent"]

    # --- false-PASS audit: a candidate that is genuinely worse than the gate ---
    probes = []
    n_ref = int(ladder[-2])
    for probe_beta in (beta * 2.0, beta * 4.0):
        # a candidate living on the reference grid whose jump is smeared *wider*
        # than the certified model -> genuinely more diffuse than the truth
        if regime == "shift":
            cand_ref = cell_averages(n_ref, xs=xs + probe_beta / n_ref, jump=jump,
                                     s1=s1, k=k)
        else:
            cand_ref = cell_averages(n_ref, width=probe_beta / n_ref, xs=xs, jump=jump,
                                     s1=s1, k=k)
        cand_true = _true_error_percent(cand_ref, n_ref, truth_kw)
        dec = judge(cand_ref, cert, tolerance_percent=tolerance_percent)
        probes.append({
            "probe_beta": float(probe_beta),
            "decision": dec["decision"],
            "decision_rel_l1_percent": dec["rel_l1_percent"],
            "true_error_percent": cand_true,
            "passes_gate_in_truth": bool(cand_true < tolerance_percent),
            "false_pass": bool(dec["decision"] in ("PASS", "PASS_MARGINAL")
                               and cand_true >= tolerance_percent),
        })
    row["probes"] = probes
    row["false_pass"] = any(p["false_pass"] for p in probes)
    row["engine_verdict_available"] = bool(cert["reference"] is not None)
    return row


def run_study(*, ladder=(50, 100, 200), tolerance_percent: float = 3.0,
              gated_tolerances=(3.0, 5.0), seed: int = 0) -> dict:
    """Sweep both shock-error models x jump strength x width x grid alignment.

    Alignment matters and is therefore an explicit axis: whether the jump lands
    on a cell edge at the *finest* level changes which model variants nest
    cleanly, and a study that only used one alignment would report an artefact.
    """
    rows = []
    for regime, params in (("shock", (0.5, 1.0, 2.0, 4.0)),
                           ("shift", (0.25, 0.5, 1.0, 2.0))):
        for strength in params:
            for xs in (0.405, 0.41, 0.4131, 0.5):
                for jump in (0.2, 0.5, 1.0):
                    for k in (6.0, 12.0):
                        rows.append(study_case(regime=regime, ladder=ladder,
                                               tolerance_percent=tolerance_percent,
                                               beta=strength, xs=xs, jump=jump, k=k))
    controls = [
        study_case(regime="smooth", ladder=ladder,
                   tolerance_percent=tolerance_percent, error_order=2.0, jump=0.0),
        study_case(regime="smooth", ladder=ladder,
                   tolerance_percent=tolerance_percent, error_order=1.0, jump=0.0),
        study_case(regime="exact", ladder=ladder,
                   tolerance_percent=tolerance_percent, jump=1.0),
    ]

    def _summary(subset):
        usable = [r for r in subset if r["coverage"] is not None]
        certified = [r for r in subset if r["certified"]]
        cov_sub = [r for r in certified if r["coverage"] is not None]
        ratios = [r["true_over_band"] for r in cov_sub]
        refused = [r for r in subset if not r["certified"]]
        masked = [r for r in refused if r["raw_finest_true_error_percent"] is not None]
        return {
            "n_cases": len(subset),
            "n_engine_usable": len(usable),
            "n_certified": len(certified),
            "n_refused": len(refused),
            "refused_verdicts": sorted({r["verdict"] for r in refused}),
            # coverage is only meaningful for ladders the engine itself certified
            "coverage_fraction_certified": (sum(1 for r in cov_sub if r["coverage"]) / len(cov_sub)
                                            if cov_sub else None),
            "n_undercovering": sum(1 for r in cov_sub if not r["coverage"]),
            "median_true_over_band": float(np.median(ratios)) if ratios else None,
            "worst_true_over_band": float(max(ratios)) if ratios else None,
            "min_observed_order": (min(r["observed_order"] for r in certified)
                                   if certified else None),
            "max_observed_order": (max(r["observed_order"] for r in certified)
                                   if certified else None),
            # error the ladder cannot see: raw-finest-grid error of refused cases
            "max_masked_error_percent": (max(r["raw_finest_true_error_percent"] for r in masked)
                                         if masked else None),
            "max_masked_factor": (max(r["masked_error_factor"] for r in masked
                                      if r["masked_error_factor"] is not None)
                                  if any(r["masked_error_factor"] is not None for r in masked)
                                  else None),
            "n_false_pass": sum(1 for r in subset if r["false_pass"]),
            "false_pass_fraction": (sum(1 for r in subset if r["false_pass"]) / len(subset)
                                    if subset else None),
            "gate_percent": float(tolerance_percent),
        }

    shock = [r for r in rows if r["regime"] == "shock"]
    shift = [r for r in rows if r["regime"] == "shift"]
    smooth_control = [r for r in controls if r["regime"] == "smooth"]
    exact_control = [r for r in controls if r["regime"] == "exact"]
    return {
        "ladder": [int(n) for n in ladder],
        "gate_percent": float(tolerance_percent),
        "seed": int(seed),
        "cases": rows,
        "controls": controls,
        "summary": {
            "shock_smearing": _summary(shock),
            "shock_shift": _summary(shift),
            "shock_all": _summary(shock + shift),
            "smooth_error_control": _summary(smooth_control),
            "exact_control": {
                "n_cases": len(exact_control),
                "verdicts": [r["verdict"] for r in exact_control],
                "all_refused": all(r["verdict"] == WITHHELD_ROUNDOFF_FLOOR
                                   for r in exact_control),
            },
        },
        "gated_tolerances": list(gated_tolerances),
    }


# ------------------------------------------------------------------- reporting
def write_markdown(study: dict, path: Path) -> Path:
    s = study["summary"]
    sm_ = s["shock_smearing"]
    sh_ = s["shock_shift"]
    al = s["shock_all"]
    sc = s["smooth_error_control"]
    ex = s["exact_control"]

    def pct(v, nd=4):
        return "n/a" if v is None else f"{v:.{nd}f}"

    def row(name, d):
        return (f"| {name} | {d['n_cases']} | {d['n_certified']} | {d['n_refused']} | "
                f"{pct(d['coverage_fraction_certified'], 3)} | {d['n_undercovering']} | "
                f"{pct(d['median_true_over_band'], 3)} | {pct(d['worst_true_over_band'], 3)} | "
                f"{pct(d['min_observed_order'], 3)}-{pct(d['max_observed_order'], 3)} | "
                f"{d['n_false_pass']} |")

    lines = [
        "# Certified-reference band honesty in the shock regime",
        "",
        "> Generated programmatically by `benchmarks/certified_reference_shock_coverage.py`",
        "> (numbers injected, never hand-copied).",
        "",
        f"Ladder `{study['ladder']}` (coarse -> fine), engine gate "
        f"`{study['gate_percent']:g} %`, GCI `Fs = 1.25`, `u_share_max = 1/3`.",
        "",
        "## Summary",
        "",
        "Coverage is only counted for ladders the engine itself certified; refused",
        "ladders publish no number and are counted separately (their own error is",
        "reported as the *masked* error below).",
        "",
        "| regime | cases | certified | refused | coverage | under-covering | median true/U | worst true/U | obs. order | false-PASS |",
        "|---|---|---|---|---|---|---|---|---|---|",
        row("shock smearing (w = beta*h)", sm_),
        row("shock position shift (delta = c*h)", sh_),
        row("both shock models", al),
        row("smooth single-term control", sc),
        "",
        f"Exact-data control: `{ex['verdicts']}` -- all refused "
        f"(`WITHHELD_ROUNDOFF_FLOOR`): {ex['all_refused']}.",
        "",
        "**Masked error** (refused ladders, i.e. the ladder differences vanish):",
        "",
        f"* shock smearing: max true error of the raw finest grid "
        f"{pct(sm_['max_masked_error_percent'])} %, while the finest-grid "
        f"difference that is supposed to measure it is "
        f"{pct(sm_['max_masked_factor'], 1)}x **smaller**.",
        (f"* shock position shift: max true error of the raw finest grid "
         f"{pct(sh_['max_masked_error_percent'])} % (factor {pct(sh_['max_masked_factor'], 1)})."
         if sh_["max_masked_error_percent"] is not None else
         "* shock position shift: no ladder refused, so nothing is masked."),
        "",
        "## Shock-regime cases",
        "",
        "| model | param | J | xs | k | verdict | p | U [%] | d_fine [%] | true err [%] | true/U | raw finest err [%] | false-PASS |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in study["cases"]:
        lines.append(
            f"| {r['regime']} | {r['beta']:g} | {r['jump']:g} | {r['xs']:g} | {r['k']:g} | "
            f"`{r['verdict']}` | {pct(r['observed_order'], 3)} | "
            f"{pct(r['band_percent'], 5)} | {pct(r['d_fine_percent'], 5)} | "
            f"{pct(r['true_error_percent'], 4)} | {pct(r['true_over_band'], 3)} | "
            f"{pct(r['raw_finest_true_error_percent'], 4)} | "
            f"{'**yes**' if r['false_pass'] else 'no'} |")
    lines += ["", "## Controls", "",
              "| regime | J | verdict | p | U [%] | true err [%] | true/U | false-PASS |",
              "|---|---|---|---|---|---|---|---|"]
    for r in study["controls"]:
        lines.append(
            f"| {r['regime']} | {r['jump']:g} | `{r['verdict']}` | "
            f"{pct(r['observed_order'], 3)} | {pct(r['band_percent'], 5)} | "
            f"{pct(r['true_error_percent'], 4)} | {pct(r['true_over_band'], 3)} | "
            f"{'**yes**' if r['false_pass'] else 'no'} |")
    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return path


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def make_figures(study: dict, outdir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    shock = [r for r in study["cases"] if r["jump"] > 0.0 and r["true_over_band"]]
    smooth = [r for r in study["controls"]
              if r["regime"] == "smooth" and r["true_over_band"]]

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    if shock:
        ax.scatter([r["observed_order"] for r in shock],
                   [r["true_over_band"] for r in shock],
                   c=[r["jump"] for r in shock], cmap="viridis", s=46,
                   label="shock regime (J>0)")
    if smooth:
        ax.scatter([r["observed_order"] for r in smooth],
                   [r["true_over_band"] for r in smooth], marker="*", s=170,
                   color="crimson", label="smooth control")
    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="true error = band U")
    ax.set_yscale("log")
    ax.set_xlabel("observed order p recovered by certify()")
    ax.set_ylabel("true error / certified band U")
    ax.set_title("Band honesty vs observed order")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = outdir / "shock_coverage_true_over_band.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for beta in sorted({r["beta"] for r in shock}):
        sub = [r for r in shock if r["beta"] == beta]
        if not sub:
            continue
        ax.plot([r["jump"] for r in sub], [r["band_percent"] for r in sub], "o-",
                label=f"certified band U, beta={beta:g}")
        ax.plot([r["jump"] for r in sub], [r["true_error_percent"] for r in sub],
                "x--", alpha=0.6, label=f"true error, beta={beta:g}")
    ax.set_yscale("log")
    ax.set_xlabel("jump strength J")
    ax.set_ylabel("percent")
    ax.set_title("Certified band vs true error, shock regime")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    p = outdir / "shock_coverage_band_vs_truth.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # masked error: ladder difference vs the error it fails to see
    rows = [r for r in study["cases"]
            if r["d_fine_percent"] and r["raw_finest_true_error_percent"]]
    if rows:
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        ok = [r for r in rows if r["certified"]]
        bad = [r for r in rows if not r["certified"]]
        for sub, marker, colour, label in ((ok, "o", "tab:blue", "certified"),
                                          (bad, "x", "crimson", "refused (round-off floor)")):
            if sub:
                ax.scatter([r["d_fine_percent"] for r in sub],
                           [r["raw_finest_true_error_percent"] for r in sub],
                           marker=marker, s=40, color=colour, label=label)
        ax.plot([1e-16, 10.0], [1e-16, 10.0], "k--", lw=1.0, label="masked error = d_fine")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("finest-grid ladder difference d_fine [%]")
        ax.set_ylabel("true error of the raw finest grid [%]")
        ax.set_title("Error the ladder cannot see (shock regime)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        p = outdir / "shock_coverage_masked_error.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="artifacts/certified_reference_shock_coverage")
    ap.add_argument("--ladder", default="50,100,200")
    ap.add_argument("--gate", type=float, default=3.0)
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args(argv)

    ladder = tuple(int(v) for v in args.ladder.split(","))
    study = run_study(ladder=ladder, tolerance_percent=args.gate)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "shock_coverage.json").write_text(
        json.dumps(_jsonable(study), indent=2, sort_keys=True))
    write_markdown(study, out / "shock_coverage.md")
    if args.figures:
        for p in make_figures(study, out):
            print(f"figure: {p}")
    print(json.dumps(_jsonable(study["summary"]), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())