"""Produce the evidence artifact and figure for CASE-DG-VISCOUS-BURGERS-COLEHOPF.

    python -m tools.viscous_burgers_colehopf

Writes ``validation/viscous_burgers_1d/results.json`` (every number the case
document and the platform report quote is read back from there, zero hand
copying) and ``docs/figures/viscous_burgers_colehopf.png``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tensordg.cases import viscous_burgers_1d as vb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "viscous_burgers_1d"
FIG = ROOT / "docs" / "figures" / "viscous_burgers_colehopf.png"

#: (cells, degree) ladder -- kept small on purpose: the diffusive step is O(h^2)
LADDER = [(8, 1), (16, 1), (32, 1), (8, 2), (16, 2), (8, 3)]


def _profile(result):
    x = result["centers"][:, None] + 0.5 * result["h"] * vb._flux_operators(
        result["degree"], vb.torch.device("cpu"), result["dtype"])[0].numpy()[None, :]
    values = result["coefficients"].numpy() @ vb._flux_operators(
        result["degree"], vb.torch.device("cpu"), result["dtype"])[2].numpy().T
    return x.reshape(-1), values.reshape(-1)


def main() -> dict:
    parameters = vb.viscous_burgers_parameters()
    runs = []
    for cells, degree in LADDER:
        result = vb.solve_viscous_burgers_1d(cells, degree)
        runs.append({
            "cells": cells,
            "degree": degree,
            "dofs": cells * (degree + 1),
            "steps": result["steps"],
            "dt": result["dt"],
            "lambda_max": result["lambda_max"],
            "relative_l2_error": vb.l2_error(result),
            "total_mass": vb.total_mass(result),
            "total_energy": vb.total_energy(result),
            "h": result["h"],
        })
        print(f"p={degree} N={cells:3d} steps={result['steps']:6d} relL2={runs[-1]['relative_l2_error']:.5%}")

    orders = {}
    for degree in (1, 2, 3):
        series = [r for r in runs if r["degree"] == degree]
        series.sort(key=lambda r: r["cells"])
        orders[str(degree)] = [
            {
                "cells": [series[i]["cells"], series[i + 1]["cells"]],
                "observed_order": math.log2(series[i]["relative_l2_error"]
                                            / series[i + 1]["relative_l2_error"]),
            }
            for i in range(len(series) - 1)
        ]

    x = (np.arange(4096) + 0.5) / 4096
    exact = vb.exact_solution(x, parameters["final_time"], parameters["nu"])
    reference = {
        "mean": float(exact.mean()),
        "max_abs_antisymmetry": float(np.max(np.abs(
            vb.exact_solution(0.5 + np.array([0.1, 0.25, 0.4]), parameters["final_time"], parameters["nu"])
            + vb.exact_solution(0.5 - np.array([0.1, 0.25, 0.4]), parameters["final_time"], parameters["nu"])))),
        "energy": float(0.5 * np.mean(exact ** 2)),
        "max_fourier_vs_quadrature": float(np.max(np.abs(
            vb.exact_solution_fourier(x[::97], parameters["final_time"], parameters["nu"], modes=256)
            - vb.exact_solution(x[::97], parameters["final_time"], parameters["nu"])))),
        "amplitude": float(np.max(np.abs(exact))),
        "layer_width_estimate": float(np.sqrt(4.0 * parameters["nu"] * parameters["final_time"])),
    }

    documented = next(r for r in runs if r["cells"] == parameters["cells"] and r["degree"] == parameters["degree"])
    payload = {
        "case_key": parameters["case_key"],
        "parameters": parameters,
        "reference": reference,
        "runs": runs,
        "observed_orders": orders,
        "documented_run": documented,
        "acceptance": {
            "tolerance": parameters["tolerance"],
            "measured": documented["relative_l2_error"],
            "passes": documented["relative_l2_error"] < parameters["tolerance"],
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")

    # ---- figure: profiles + convergence
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    left.plot(x, exact, "k-", lw=2.0, label="Cole-Hopf exact")
    for degree, style in ((1, "o--"), (2, "s-"), (3, "^:")):
        result = vb.solve_viscous_burgers_1d(16 if degree < 3 else 8, degree)
        px, pu = _profile(result)
        left.plot(px, pu, style, ms=3.0,
                  label=f"DG P{degree}, N={result['ncells']} (rel $L^2$ {vb.l2_error(result):.2%})")
    left.set_xlabel("x"); left.set_ylabel("u(x, T)")
    left.set_title(f"periodic viscous Burgers, nu={parameters['nu']}, T={parameters['final_time']}")
    left.legend(fontsize=7); left.grid(alpha=0.3)

    for degree, style in ((1, "o-"), (2, "s-"), (3, "^-")):
        series = sorted([r for r in runs if r["degree"] == degree], key=lambda r: r["h"])
        right.loglog([r["h"] for r in series], [r["relative_l2_error"] for r in series],
                     style, ms=4, label=f"DG P{degree}")
    hs = np.array([0.125, 0.02])
    right.loglog(hs, 3e-3 * (hs / 0.125) ** 2, "k--", lw=0.8, label="slope 2")
    right.loglog(hs, 3e-3 * (hs / 0.125) ** 3, "k:", lw=0.8, label="slope 3")
    right.axhline(parameters["tolerance"], color="r", ls="--", lw=0.8, label="3% tolerance")
    right.set_xlabel("h = 1/N"); right.set_ylabel("relative $L^2$ error")
    right.set_title("h-refinement vs Cole-Hopf"); right.legend(fontsize=7); right.grid(alpha=0.3, which="both")

    FIG.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIG, dpi=150)
    plt.close(figure)
    print(f"wrote {OUT / 'results.json'} and {FIG}")

    # markdown table for the case document (pasted verbatim: zero hand copying)
    print("\n| p | N | dofs | steps | dt | rel. L2 error | mass | energy |")
    print("|---|---|---|---|---|---|---|---|")
    for r in runs:
        print(f"| {r['degree']} | {r['cells']} | {r['dofs']} | {r['steps']} | {r['dt']:.3e} | "
              f"{r['relative_l2_error']:.5%} | {r['total_mass']:+.1e} | {r['total_energy']:.8f} |")
    print("\norders:")
    for degree, entries in orders.items():
        print(f"  P{degree}: " + ", ".join(
            f"N{e['cells'][0]}->{e['cells'][1]}: {e['observed_order']:.2f}" for e in entries))
    return payload


if __name__ == "__main__":
    main()