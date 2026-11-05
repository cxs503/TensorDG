"""Build the PR #100 evidence figures/table fragments from recomputed residuals.

Outputs (all under the work tree):

* ``docs/figures/tensordg-pr100-wall-pin.png`` -- two panels: cold-wall vs
  interior energy residual under refinement (log-log, with O(h) reference), and
  the entry-point temperature profile against the exact conduction line;
* ``validation/ns_heat_wall_closure/pr100_figure_data.json`` -- the plotted
  numbers, so every digit in the report is machine-generated;
* ``validation/ns_heat_wall_closure/pr100_table.md`` -- markdown table fragment
  injected into the report doc by ``--inject``.

Run: ``PYTHONPATH=. python validation/ns_heat_wall_closure/pr100_figure.py --inject docs/reports/tensordg-pr100-ci-red-repair.md``
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensordg import ns_dg_2d as ns
from tests.test_ns_heat_wall_closure import (
    PRANDTL,
    T_BOTTOM,
    T_TOP,
    _residual,
    _setup,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPRO = HERE / "pr100_repro.json"
DATA = HERE / "pr100_figure_data.json"
TABLE = HERE / "pr100_table.md"
FIGURE = ROOT / "docs" / "figures" / "tensordg-pr100-wall-pin.png"


def _sweep():
    rows = []
    for ny in (8, 16, 32, 64):
        residual = _residual(_setup(ny)).abs().amax(dim=(2, 3))
        rows.append({
            "ny": ny,
            "wall": float(residual[:, 0, 3].max()),
            "interior": float(residual[:, 1:-1, 3].max()),
        })
    for prev, cur in zip(rows, rows[1:]):
        cur["wall_ratio"] = prev["wall"] / cur["wall"]
        cur["interior_ratio"] = prev["interior"] / cur["interior"]
    return rows


def _entry_profile():
    """Temperature along y at the first x-cell from the shipped entry point."""
    _, _, coeff, _ = ns.solve_heat_conduction_dg_2d(
        ncells=(2, 8), final_time=0.5, dtype=torch.float64)
    fields = ns.cell_fields(coeff)
    return [float(value) for value in (fields["pressure"] / fields["density"])[0].tolist()]


def _plot(rows, entry):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ny = [row["ny"] for row in rows]
    wall = [row["wall"] for row in rows]
    interior = [row["interior"] for row in rows]
    ax_a.loglog(ny, wall, "o-", color="#c0392b", label="cold-wall row $\\max|R_E|$")
    ax_a.loglog(ny, interior, "s-", color="#2c7fb8", label="interior $\\max|R_E|$")
    reference = [interior[0] * (ny[0] / n) for n in ny]
    ax_a.loglog(ny, reference, "k--", lw=1.0, label="$O(h)$ reference")
    for row in rows[1:]:
        ax_a.annotate(f"{row['wall_ratio']:.4f}", (row["ny"], row["wall"]),
                      textcoords="offset points", xytext=(6, 6), fontsize=8,
                      color="#c0392b")
    ax_a.set_xlabel("cells in y ($p=1$)")
    ax_a.set_ylabel(r"$\max|R_E|$ (exact conduction state)")
    ax_a.set_title("cold wall stagnates, interior halves")
    ax_a.grid(True, which="both", alpha=0.3)
    ax_a.legend(fontsize=8)

    profile = entry.get("theta_profile") or _entry_profile()
    y_cells = len(profile)
    y = [(i + 0.5) / y_cells for i in range(y_cells)]
    exact = [T_BOTTOM + (T_TOP - T_BOTTOM) * value for value in y]
    ax_b.plot(y, profile, "o-", ms=4, color="#2c7fb8", label="DG entry point, $t=0.5$")
    ax_b.plot(y, exact, "k--", lw=1.2, label="exact conduction")
    offset = max(abs(a - b) for a, b in zip(profile, exact))
    ax_b.set_xlabel("y")
    ax_b.set_ylabel(r"$\theta = p/\rho$")
    ax_b.set_title(f"entry point completes (max |$\\Delta\\theta$| = {offset:.3e})")
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(fontsize=8)

    fig.suptitle(
        f"TensorDG PR #100: isothermal-wall pin audit "
        f"($\\mu$={ns.MU if hasattr(ns, 'MU') else 0.1}, Pr={PRANDTL}, "
        f"$T$={T_BOTTOM}/{T_TOP})",
        fontsize=10,
    )
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=160)
    plt.close(fig)
    return offset


def _table(rows, ratios):
    lines = [
        "| ny | cold-wall $\\max|R_E|$ | wall ratio vs prev | interior $\\max|R_E|$ | interior ratio vs prev |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        wall_ratio = f"{row['wall_ratio']:.4f}" if "wall_ratio" in row else "--"
        interior_ratio = f"{row['interior_ratio']:.4f}" if "interior_ratio" in row else "--"
        lines.append(
            f"| {row['ny']} | {row['wall']:.4e} | {wall_ratio} | "
            f"{row['interior']:.4e} | {interior_ratio} |"
        )
    lines += [
        "",
        f"Stale pin `wall_ratio(16/32) < 1.1` against measured "
        f"**{ratios['wall_ratio_16_32']:.10f}** -> fails by "
        f"{(ratios['wall_ratio_16_32'] / 1.1 - 1) * 100:.2f} %.",
        f"Repaired pins (`< 1.2` at 16/32, `< 1.1` at 32/64, monotone, floor "
        f"> 1e-2) with measured {ratios['wall_ratio_32_64']:.10f} -> all hold.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inject", type=Path, help="report doc to inject the table into")
    args = parser.parse_args()

    repro = json.loads(REPRO.read_text())
    rows = _sweep()
    offset = _plot(rows, repro["entry_point"])
    DATA.write_text(json.dumps({"residual_sweep": rows, "entry_offset": offset}, indent=2) + "\n")
    table = _table(rows, repro["ratios"])
    TABLE.write_text(table + "\n")
    print(table)
    if args.inject is not None:
        doc = args.inject.read_text()
        marker = "<!-- TABLE:wall-residual -->"
        assert marker in doc, f"marker missing in {args.inject}"
        args.inject.write_text(doc.replace(marker, table))
        print(f"injected into {args.inject}")
    print(f"figure -> {FIGURE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())