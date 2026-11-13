"""Figure for the p=5..8 measurement-floor study (CASE-DG-PREFINEMENT-EXPONENTIAL).

Reads the two artifacts and draws both panels; no number is typed by hand, so the
figure cannot drift from the data:

* left  -- L2 error vs polynomial degree: the published p=1..6 ladder, the
  degrees re-measured with time-step certification (``measurement_floor.json``),
  the constant-sigma extrapolation from p=6, and the float64 measurement
  floor with its ``SNR_MIN`` margin;
* right -- local decay rate ``sigma(p) = ln(e(p-1)/e(p))`` against the
  ``0.5..2`` sigma band, which shows where exponential convergence holds and
  where the plateau (roundoff / temporal floor) takes over.

Run::

    python -m tools.plot_floor_high_degree
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "docs" / "benchmarks" / "results" / "p-refinement-exponential"
PRIMARY = RESULTS / "p_refinement_results.json"
FLOOR = RESULTS / "measurement_floor.json"
OUT = RESULTS / "measurement_floor_p5_p8.png"


def build(out: Path = OUT) -> Path:
    primary = json.loads(PRIMARY.read_text())
    floor_art = json.loads(FLOOR.read_text())

    p_pub = [int(p) for p in primary["results"]["degrees"]]
    e_pub = [float(e) for e in primary["results"]["errors"]]
    sigma = float(primary["results"]["diagnostics"]["lsq_decay_rate"])

    levels = sorted(floor_art["levels"], key=lambda lv: int(lv["degree"]))
    p_meas = [int(lv["degree"]) for lv in levels]
    e_meas = [float(lv["l2_extended_longdouble"]) for lv in levels]
    spread = [float(lv["path_spread_rel"]) * float(lv["l2_extended_longdouble"])
              for lv in levels]
    floor = float(floor_art["floor"]["value"])
    snr_min = float(floor_art["floor"]["snr_min"])

    p_ref = int(floor_art["extrapolation_check"]["reference_degree"])
    e_ref = float(floor_art["extrapolation_check"]["reference_error"])
    p_pred = list(range(p_ref, max(max(p_pub), max(p_meas)) + 2))
    e_pred = [e_ref * math.exp(-sigma * (p - p_ref)) for p in p_pred]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    ax.semilogy(p_pub, e_pub, "o-", color="C0", label="published ladder (p=1..6)")
    ax.errorbar(p_meas, e_meas, yerr=spread, fmt="s", color="C1", capsize=4,
                label="re-measured, time-step certified")
    ax.semilogy(p_pred, e_pred, "--", color="C2",
                label=fr"$\sigma={sigma:.3f}$ extrapolation from p={p_ref}")
    ax.axhline(floor, ls=":", color="0.4",
               label=fr"float64 floor $\epsilon_{{64}}\|u_h\|$ = {floor:.2e}")
    ax.axhline(snr_min * floor, ls="-.", color="0.6", lw=1.0,
               label=fr"measurable margin ({snr_min:g}$\times$ floor)")
    ax.set_xlabel("polynomial degree $p$ (ncells=16 fixed)")
    ax.set_ylabel(r"$L_2$ error vs analytic solution")
    ax.set_title("p-refinement, smooth $\\sin(2\\pi x)$: exponential convergence")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper right")

    ladder = list(zip(p_pub, e_pub)) + [(p, e) for p, e in zip(p_meas, e_meas)
                                        if p not in p_pub]
    ladder.sort()
    degs = [p for p, _ in ladder]
    local = [math.log(ladder[i - 1][1] / ladder[i][1]) / (degs[i] - degs[i - 1])
             for i in range(1, len(ladder))]
    ax2.plot(degs[1:], local, "o-", color="C3", label=r"local $\sigma(p)$")
    ax2.axhspan(0.5 * sigma, 2.0 * sigma, color="C3", alpha=0.12,
                label=fr"$0.5$-$2\times\sigma_{{lsq}}$ band ({0.5 * sigma:.2f}-{2 * sigma:.2f})")
    ax2.axhline(sigma, ls="--", color="C2", label=fr"$\sigma_{{lsq}}={sigma:.3f}$ (p=1..6 fit)")
    ax2.set_xlabel("polynomial degree $p$")
    ax2.set_ylabel(r"local decay rate $\ln(e_{p-1}/e_p)$")
    ax2.set_title("where the exponential law holds")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"figure written: {out}")
    return out


if __name__ == "__main__":
    build()