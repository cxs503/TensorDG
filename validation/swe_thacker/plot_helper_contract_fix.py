"""Figure for the SWE well-balanced helper-contract fixes.

Programmatically compares the *old* helper behaviour (hard-coded 1e-10 clamp in
``estimate_well_balanced_error_bound``; variance-based acceptance in
``validate_well_balanced_setup``) with the *fixed* behaviour on the same inputs,
so the two panels of the figure contain no hand-copied numbers.

Panel (a): theoretical representation estimate of the free-surface error versus
           the number of cells, P1 and P2, for an O(1) bed curvature, together
           with the flat 1e-10 value the old code returned for every mesh, and
           the measured round-off floor of the solver quoted in the docs.
Panel (b): lake-at-rest acceptance test as a function of the free-surface drift
           amplitude: old criterion (variance vs 1e-10) versus new criterion
           (sup norm vs 1e-10); the shaded band is the region the old test
           wrongly accepted.

Run:  python validation/swe_thacker/plot_helper_contract_fix.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from tensordg.swe_1d import (
    estimate_well_balanced_error_bound,
    validate_well_balanced_setup,
)

OUT = Path("docs/figures/swe_wb_helper_contract.png")

# Round-off floor actually measured for the well-balanced path and reported in
# docs/tensordg-swe-hydrostatic-wellbalanced.md (P1, curved bed).
ROUNDOFF_FLOOR = 2.3e-10
OLD_CLAMP = 1e-10
TOL = 1e-10


def panel_a(ax):
    ncells = [8, 16, 32, 64, 128, 256, 512, 1024]
    for degree, style in ((1, "-o"), (2, "-s")):
        est = [
            estimate_well_balanced_error_bound(
                n, degree, 1.0, bed_curvature=1.0)["eta_error_estimate"]
            for n in ncells
        ]
        ax.loglog(ncells, est, style, ms=4,
                  label=f"P{degree} representation estimate (fixed)")
        # The old code returned min(estimate, 1e-10): for every mesh shown the
        # estimate is far above the clamp, so the *whole* curve above was
        # replaced by the flat red line.
        assert est[0] > 10.0 * OLD_CLAMP, (
            "on the coarsest mesh the clamp must have replaced the estimate")
    ax.axhline(OLD_CLAMP, color="crimson", ls="--", lw=1.4,
               label="old hard-coded clamp 1e-10 m")
    ax.axhline(ROUNDOFF_FLOOR, color="0.4", ls=":", lw=1.4,
               label=f"measured round-off floor {ROUNDOFF_FLOOR:.1e} m")
    ax.set_xlabel("cells N")
    ax.set_ylabel(r"$\eta$ error estimate [m]")
    ax.set_title("(a) error-bound helper: estimate vs old clamp")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")


def panel_b(ax):
    eta0 = 2.0
    amplitudes = torch.logspace(-9.0, -4.0, 60, dtype=torch.float64)

    def zb(xx):
        return 0.3 * torch.cos(2.0 * math.pi * xx)

    old_accept, new_accept = [], []
    for a in amplitudes:
        amp = float(a)

        def initial_h(xx, amp=amp):
            return (eta0 - zb(xx) + amp * torch.sin(2.0 * math.pi * xx))

        res = validate_well_balanced_setup(
            initial_h, lambda xx: torch.zeros_like(xx), zb)
        new_accept.append(1.0 if res["is_valid"] else 0.0)
        # old criterion, reproduced from the pre-fix source line
        old_accept.append(1.0 if res["eta_variance"] < TOL else 0.0)

    amp = amplitudes.numpy()
    ax.loglog(amp, [max(v, 1e-30) for v in old_accept], drawstyle="steps-post",
              color="crimson", lw=1.6, label="old: variance < 1e-10 (accept)")
    ax.loglog(amp, [max(v, 1e-30) for v in new_accept], drawstyle="steps-post",
              color="seagreen", lw=1.6, label="new: max$|\\eta-\\langle\\eta\\rangle|$ < 1e-10")
    ax.axvline(TOL, color="0.3", ls=":", lw=1.2,
               label="stated tolerance 1e-10 m")
    ax.fill_betweenx([0.7, 1.3], TOL, math.sqrt(2.0 * TOL), color="crimson",
                     alpha=0.15)
    ax.annotate("old false-positive band\n(1e-10 m … 1.4e-5 m drift accepted)",
                xy=(3e-8, 1.02), xytext=(2e-9, 0.55), fontsize=7, ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_ylim(0.4, 1.6)
    ax.set_xlabel(r"free-surface drift amplitude $a$ in $a\sin(2\pi x)$ [m]")
    ax.set_ylabel("accept (1) / reject (0)")
    ax.set_title("(b) setup-validation helper: acceptance vs drift")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panel_a(axes[0])
    panel_b(axes[1])
    fig.suptitle("SWE hydrostatic well-balanced helpers: contract fixes", fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()