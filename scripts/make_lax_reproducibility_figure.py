"""Render the CASE-LAX-DG reproducibility-band figures from the artifact."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.lax_case_pool import DIAPHRAGM, FINAL_TIME, LEFT, RIGHT, exact_reference

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "docs/benchmarks/results/lax_reproducibility_band"
LEDGER_L2 = 3.1155618811580767e-02


def main():
    payload = json.loads((ART / "metrics.json").read_text())
    rows = [r for r in payload["rows"] if r.get("drift_l1") is not None]
    local = sorted({r["stack"] for r in rows})
    payload_fn = plt.figure(figsize=(11, 4.2))
    ax = payload_fn.add_subplot(1, 2, 1)
    for stack in local:
        ladder = sorted((r for r in rows if r["stack"] == stack and r["torch_threads"] == 1),
                        key=lambda r: r["steps"])
        steps = [r["steps"] for r in ladder]
        ax.loglog(steps, [r["drift_l1"] for r in ladder], "o-",
                  label=f"L1 rho drift ({stack})")
        ax.loglog(steps, [abs(r["l2_density"] / LEDGER_L2 - 1.0) for r in ladder], "s--",
                  label="L2 rho drift vs ledger reading")
    ax.axhline(1e-6, color="crimson", lw=1.2, ls=":",
               label="existing reuse gate tolerance 1e-6")
    ax.set_xlabel("time steps of the leg")
    ax.set_ylabel("relative drift vs committed artifact")
    ax.set_title("stack drift grows with run length\n(L1 breaks the 1e-6 gate, L2 does not)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax2 = payload_fn.add_subplot(1, 2, 2)
    points, numerical = _rebuild_profiles(400, 3, payload)
    exact = exact_reference(points)
    ax2.plot(points.ravel(), exact["density"].ravel(), "k-", lw=1.4, label="exact Riemann")
    ax2.plot(points.ravel(), numerical.ravel(), "C1--", lw=0.9, label="DG p3, N=400 (this stack)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("density rho")
    ax2.set_title("p3 N=400: the two stacks are visually identical\n(they differ in the 5th digit)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    out = ART / "figures/lax_reproducibility_band.png"
    payload_fn.tight_layout()
    payload_fn.savefig(out, dpi=140)
    print(out)


def _rebuild_profiles(cells, degree, payload):
    """Rebuild the Gauss-node abscissae and read the committed density field."""
    cache = ART / "profiles" / f"p{degree}.N{cells}_density.npz"
    if not cache.exists():
        raise SystemExit(f"missing profile cache {cache}")
    data = np.load(cache)
    density = data["density"]
    nodes, _ = np.polynomial.legendre.leggauss(10)
    dx = 1.0 / cells
    centres = (np.arange(cells) + 0.5) * dx
    points = centres[:, None] + 0.5 * dx * nodes[None, :]
    assert points.shape == density.shape, (points.shape, density.shape)
    return points, density


if __name__ == "__main__":
    main()
