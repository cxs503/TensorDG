"""Driver for the DG elastodynamics dispersion / dissipation spectrum.

Anchor: ``CASE-DG-DALEMBERT-ELASTIC-1D`` (the *scheme* dispersion counterpart of
that case's profile error gate).

    python -m benchmarks.elastic_dispersion [--out DIR] [--no-figures]

Writes ``elastic_dispersion.json`` (+ two PNGs) under
``docs/benchmarks/results/elastic-dispersion/`` by default.  Every number in the
JSON is computed here from the shipped operator -- nothing is hand-copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from tensordg.elastic_dispersion import (
    DEGREES,
    analyse,
    dominant_theta,
    exponent_details,
    phase_and_dissipation,
    resolution_table,
    symbol_union_error,
    p0_symbol_error,
)
from tensordg.elastic_1d import (
    dalembert_pulse,
    relative_error,
    solve_elastic_1d,
    stress_profile,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "docs" / "benchmarks" / "results" / "elastic-dispersion"

#: Configuration of the d'Alembert case the anchor grades at <2%.
CASE = {"ncells": 200, "degree": 3, "cfl": 0.3, "final_time": 0.2,
        "length": 1.0, "width": 0.05}


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:                                    # pragma: no cover
        return "unknown"


def case_dispersion_budget() -> dict:
    """Actual d'Alembert case error vs the dispersion expected at its theta.

    The dominant wavenumber of the initial Gaussian is ``theta_dom = 2 h / w``
    (see :func:`tensordg.elastic_dispersion.dominant_theta`).  The phase error
    of the *scheme* at that wavenumber is compared with the error the case
    actually achieves: if the scheme error is orders below the achieved error,
    the profile gate is not dispersion-limited and raising the degree further
    would not buy accuracy (it would only shrink the projection / time-stepping
    contribution).
    """
    centers, coeff, meta = solve_elastic_1d(
        ncells=CASE["ncells"], degree=CASE["degree"],
        final_time=CASE["final_time"], length=CASE["length"],
        cfl=CASE["cfl"], bc="periodic")
    x, v, s = stress_profile(centers, coeff)
    _, s_exact = dalembert_pulse(x, meta["final_time"])
    achieved = relative_error(x, s, s_exact)

    h = CASE["length"] / CASE["ncells"]
    theta = dominant_theta(CASE["width"], h)
    data = phase_and_dissipation(CASE["degree"], [theta])
    return {
        "configuration": dict(CASE),
        "elements_per_pulse_width": CASE["width"] / h,
        "theta_dominant": theta,
        "measured_relative_l2_stress_error": achieved,
        "scheme_phase_error_at_theta_dominant": float(
            abs(data["phase_error"][0])),
        "scheme_dissipation_at_theta_dominant": float(
            data["dissipation"][0]),
        "dispersion_share_of_error": float(
            abs(data["phase_error"][0]) / achieved) if achieved > 0 else None,
        "time_steps": meta["steps"],
        "dt": meta["dt"],
    }


def _exponent_rows(degrees=DEGREES) -> list[dict]:
    rows = []
    for p in degrees:
        rows.append({
            "degree": p,
            "phase": exponent_details(p, "phase_error"),
            "dissipation": exponent_details(p, "dissipation"),
            "resolution": resolution_table(p),
            "symbol_union_error": symbol_union_error(p),
            "p0_anchor_error": p0_symbol_error() if p == 0 else None,
        })
    return rows


def _figures(out: Path, rows: list[dict]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thetas = np.geomspace(1e-3, math.pi, 120)
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for p in (0, 1, 2, 3, 4):
        data = phase_and_dispersion = phase_and_dissipation(p, thetas)
        axes[0].loglog(thetas, np.abs(data["phase_error"]) + 1e-17, label=f"p={p}")
        axes[1].loglog(thetas, np.asarray(data["dissipation"]) + 1e-17,
                       label=f"p={p}")
    for ax, title in zip(axes, ("phase-speed error |c_num/c - 1|",
                                "dissipation -Re(lambda) h / c")):
        ax.set_xlabel("theta = k h")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].axhline(1e-13, color="k", ls=":", lw=1)
    axes[0].annotate("double-precision floor", (2e-2, 2e-13), fontsize=7)
    fig.suptitle("DG elastodynamics operator spectrum (v-sigma system, "
                 "characteristic upwind flux)")
    path = out / "spectrum.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path.name)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ppw1 = [r["resolution"]["ppw_0.01"] for r in rows]
    ppw2 = [r["resolution"]["ppw_0.02"] for r in rows]
    deg = [r["degree"] for r in rows]
    ax.plot(deg, ppw1, "o-", label="1% phase error")
    ax.plot(deg, ppw2, "s--", label="2% phase error")
    ax.set_yscale("log")
    ax.set_xlabel("polynomial degree p")
    ax.set_ylabel("points per wavelength (budget)")
    ax.set_title("resolution budget, 1-D elastic DG")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    path = out / "resolution_budget.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path.name)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = _exponent_rows()
    report = analyse()
    budget = case_dispersion_budget()
    figures = [] if args.no_figures else _figures(out, rows)

    verdict = "PASS"
    for row in rows:
        p = row["degree"]
        if row["symbol_union_error"] > 1e-12:
            verdict = "FAIL"
        phase_exp = row["phase"]["exponent"]
        if math.isnan(phase_exp):
            # below the double-precision detection floor: report as
            # "not measurable", never as a pass or a failure
            row["phase"]["measurable"] = False
            row["phase"]["reason"] = (
                "phase error at theta<=1 sits below the 1e-10 noise floor; "
                "the exponent cannot be resolved in float64")
        else:
            row["phase"]["measurable"] = True
            if p >= 1 and not phase_exp > 2 * p + 1.0:   # floor 2p+1
                verdict = "FAIL"
        if not (row["dissipation"]["exponent"] > 2 * p + 1.0):
            verdict = "FAIL"
    if budget["measured_relative_l2_stress_error"] > 0.02:
        verdict = "FAIL"

    payload = {
        "schema": "tensordg-elastic-dispersion/1",
        "anchor": "CASE-DG-DALEMBERT-ELASTIC-1D",
        "git_sha": _git_sha(),
        "verdict": verdict,
        "convention": "select_sign: |Im| for phase, -Re for dissipation at "
                      "assembled p=0 spectrum",
        "rows": rows,
        "analyse": report,
        "case_budget": budget,
        "figures": figures,
        "notes": [
            "Exponents are measured on the shipped operator, window "
            "theta in [0.15, 1.0]; p=4 phase error sits at the "
            "double-precision floor (~1e-13) and is reported as not "
            "measurable rather than extrapolated.",
            "Independent anchor: the assembled p=0 symbol reproduces the "
            "characteristic-upwind finite-volume symbol in closed form.",
        ],
    }
    text = json.dumps(payload, indent=2, sort_keys=False, default=float)
    target = out / "elastic_dispersion.json"
    target.write_text(text)
    payload["result_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    print(f"wrote {target}  verdict={verdict}")
    print(f"  sha256={payload['result_sha256']}")
    for row in rows:
        print(f"  p={row['degree']}: phase exp={row['phase']['exponent']:.2f} "
              f"(n={row['phase']['points_used']})  "
              f"diss exp={row['dissipation']['exponent']:.2f}  "
              f"ppw@1%={row['resolution']['ppw_0.01']}")
    print(f"  case: error={budget['measured_relative_l2_stress_error']:.3e} "
          f"theta_dom={budget['theta_dominant']:.3f} "
          f"scheme phase err={budget['scheme_phase_error_at_theta_dominant']:.3e}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())