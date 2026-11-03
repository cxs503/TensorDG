"""Generate the matrix-free SIPG verification evidence.

This script is the reproducible driver behind
``docs/verify_matrixfree_backend.md``.  It re-runs the verification matrix for
the two ``matrix-free`` (PCG) backends and writes machine-readable evidence to
``validation/matrixfree/verification.json`` plus a figure.  Every number in the
report is read back from that JSON file -- nothing is transcribed by hand.

Usage:
    PYTHONPATH=. python tools/verify_matrixfree_backend.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensordg import sipg_matrixfree as mf  # noqa: E402
from tensordg.heat_1d import (  # noqa: E402
    heat_error,
    solve_heat_1d,
)
from tensordg.poisson_2d import (  # noqa: E402
    poisson_error_2d,
    solve_poisson_2d,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "validation" / "matrixfree" / "verification.json"
OUT_FIG = ROOT / "docs" / "figures" / "matrixfree_verification.png"


def poisson_study() -> dict:
    rows = []
    for degree, meshes in ((1, (4, 8, 16)), (2, (4, 8))):
        for ncells in meshes:
            x, y, coeff, meta = solve_poisson_2d(
                ncells=ncells, degree=degree, backend="matrix-free",
            )
            errors = poisson_error_2d(x, y, coeff)
            rows.append({
                "degree": degree,
                "ncells": ncells,
                "dofs": meta["dofs"],
                "relative_l2": errors["relative_l2"],
                "absolute_l2": errors["absolute_l2"],
                "cg_iterations": meta["cg_iterations"],
                "cg_converged": meta["cg_converged"],
                "cg_relative_residual": meta["cg_relative_residual"],
                "cg_rtol": meta["cg_rtol"],
            })
        sub = [r for r in rows if r["degree"] == degree]
        for coarse, fine in zip(sub, sub[1:]):
            ratio = fine["ncells"] / coarse["ncells"]
            fine["order_vs_previous"] = math.log(
                coarse["relative_l2"] / fine["relative_l2"], ratio,
            )
        for row in sub:
            row.setdefault("order_vs_previous", None)
    return {"rows": rows}


def tolerance_study() -> dict:
    rows = []
    reference = None
    for rtol in (1e-6, 1e-10, 1e-13):
        x, y, coeff, meta = solve_poisson_2d(
            ncells=8, degree=1, backend="matrix-free", rtol=rtol,
        )
        scale = float(coeff.abs().max())
        if reference is None:
            reference = coeff.clone()
        rows.append({
            "rtol": rtol,
            "cg_iterations": meta["cg_iterations"],
            "cg_converged": meta["cg_converged"],
            "cg_relative_residual": meta["cg_relative_residual"],
            "relative_l2": poisson_error_2d(x, y, coeff)["relative_l2"],
            "coeff_deviation_from_first": float(
                (coeff - reference).abs().max()
            ) / scale,
        })
    return {"rows": rows}


def budget_probe() -> dict:
    probes = []
    for maxiter in (1, 2, 10, 50):
        x, y, coeff, meta = solve_poisson_2d(
            ncells=8, degree=1, backend="matrix-free",
            rtol=1e-14, maxiter=maxiter,
        )
        probes.append({
            "maxiter": maxiter,
            "cg_iterations": meta["cg_iterations"],
            "cg_converged": meta["cg_converged"],
            "cg_relative_residual": meta["cg_relative_residual"],
            "relative_l2": poisson_error_2d(x, y, coeff)["relative_l2"],
        })
    return {"rows": probes}


def heat_study() -> dict:
    final_time = 0.1
    rows = []
    for ncells in (8, 16):
        steps = 20
        centers, coeff, meta = mf.solve_heat_1d_matrixfree(
            ncells=ncells, degree=2, final_time=final_time,
            time_steps=steps, rtol=1e-10,
        )
        centers_dense, coeff_dense, meta_dense = solve_heat_1d(
            ncells=ncells, degree=2, final_time=final_time, time_steps=steps,
        )
        rows.append({
            "ncells": ncells,
            "degree": 2,
            "dofs": meta["dofs"],
            "steps": steps,
            "dt": meta["dt"],
            "relative_l2_vs_analytic": heat_error(
                centers, coeff, final_time,
            )["relative_l2"],
            "total_cg_iterations": meta["total_cg_iterations"],
            "cg_iterations_per_step": meta["total_cg_iterations"] / steps,
            "cg_all_converged": meta["cg_all_converged"],
            "coeff_max_diff_vs_dense": float(
                (coeff - coeff_dense).abs().max()
            ),
            "dense_relative_l2_vs_analytic": heat_error(
                centers_dense, coeff_dense, final_time,
            )["relative_l2"],
        })
    return {"final_time": final_time, "rows": rows}


def plot(poisson: dict, tolerance: dict, heat: dict) -> None:
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    for degree, style in ((1, "o-"), (2, "s--")):
        rows = [r for r in poisson["rows"] if r["degree"] == degree]
        h = [1.0 / r["ncells"] for r in rows]
        err = [r["relative_l2"] for r in rows]
        ax.loglog(h, err, style, label=f"matrix-free P{degree}")
        if len(h) >= 2:
            h0, e0 = h[0], err[0]
            order = 2 if degree == 1 else 3
            ax.loglog(
                h, [e0 * (hi / h0) ** order for hi in h], ":",
                color=ax.lines[-1].get_color(), alpha=0.6,
                label=f"slope {order}",
            )
    ax.set_xlabel("cell size h")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.set_title("Poisson vs manufactured solution")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    rows = tolerance["rows"]
    ax.semilogx(
        [r["rtol"] for r in rows], [r["cg_iterations"] for r in rows], "o-",
    )
    ax.set_xlabel("requested CG rtol")
    ax.set_ylabel("CG iterations")
    ax.set_title("8x8 P1: iterations vs tolerance")
    ax.grid(True, which="both", alpha=0.3)
    for row in rows:
        ax.annotate(
            f"err {row['relative_l2']:.2e}",
            (row["rtol"], row["cg_iterations"]),
            textcoords="offset points", xytext=(6, 6), fontsize=8,
        )

    ax = axes[2]
    rows = heat["rows"]
    labels = [f"n={r['ncells']}" for r in rows]
    positions = range(len(rows))
    ax.bar(
        [p - 0.2 for p in positions],
        [r["relative_l2_vs_analytic"] for r in rows], width=0.4,
        label="matrix-free vs analytic",
    )
    ax.bar(
        [p + 0.2 for p in positions],
        [r["dense_relative_l2_vs_analytic"] for r in rows], width=0.4,
        label="dense vs analytic",
    )
    ax.axhline(0.03, color="k", linestyle="--", linewidth=1, label="3% gate")
    ax.set_yscale("log")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"relative $L^2$ error at $t=0.1$")
    ax.set_title("Heat 1-D (P2, CN, 20 steps)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("TensorDG matrix-free SIPG verification (analytic references)")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=140)
    plt.close(fig)


def main() -> None:
    evidence = {
        "generated_by": "tools/verify_matrixfree_backend.py",
        "backend": "matrix-free",
        "dtype": "float64",
        "poisson": poisson_study(),
        "tolerance": tolerance_study(),
        "budget": budget_probe(),
        "heat": heat_study(),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(evidence, indent=2) + "\n")
    plot(evidence["poisson"], evidence["tolerance"], evidence["heat"])
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUT_FIG.relative_to(ROOT)}")
    for row in evidence["poisson"]["rows"]:
        order = row["order_vs_previous"]
        order_text = "   n/a" if order is None else f"{order:6.3f}"
        print(
            f"  Poisson P{row['degree']} n={row['ncells']:3d} "
            f"err={row['relative_l2']:.3e} order={order_text} "
            f"cg_it={row['cg_iterations']:4d} conv={row['cg_converged']}"
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    main()