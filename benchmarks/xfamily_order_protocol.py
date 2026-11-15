"""Cross-family convergence-order experiment on a problem with an exact answer.

Purpose
-------
Exercise :mod:`tensordg.xfamily_order` on families whose orders are *known in
advance*, so the protocol's central prediction -- "the observed cross-family
order equals the order of the less accurate family" -- can be falsified rather
than assumed.  This is the mechanism that the anchor
``GAP-DIFFLBM-XSOLVER-ORDER`` relies on when it replaces a compare-to-table gate
with a cross-solver gate.

Model problem (deliberately the same one the SIPG benchmark already uses)::

    -Delta u = 2*pi^2 sin(pi x) sin(pi y)   on (0,1)^2,   u = 0 on the boundary

Families
--------
``dg_p1`` / ``dg_p2`` / ``dg_p3``
    symmetric interior-penalty DG from :mod:`tensordg.poisson_2d`, sampled at
    the shared node set with a jump-averaged trace (the natural DG field
    evaluation at an element face).
``fd2``
    cell-node 5-point finite differences (2nd order).
``fd4``
    9-point compact "Mehrstellen" finite differences (4th order):
    ``L9 u = (1 - h^2/12 D)(-F)`` with ``L9 = (4*sum_nb + sum_diag - 20)/(6h^2)``
    which is ``D(1 + h^2 D/12)`` by symbol analysis, hence 4th order.

Usage
-----
    PYTHONPATH=. python benchmarks/xfamily_order_protocol.py            # full run
    PYTHONPATH=. python benchmarks/xfamily_order_protocol.py --quick    # 2 grids

Artifacts are written to ``outputs/xfamily_order_protocol/``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from tensordg import solve_poisson_2d
from tensordg.convergence import observed_order, pairwise_orders
from tensordg.xfamily_order import (
    cross_family_difference,
    cross_family_verdict,
    half_gate_budget,
    max_point_error,
    percent_of,
    required_resolution,
)

OUT = Path(__file__).resolve().parent.parent / "outputs" / "xfamily_order_protocol"

# Gate copied from the anchor's acceptance text (feature_ledger
# GAP-DIFFLBM-XSOLVER-ORDER): max-point < 3% and observed order p >= 1.8.
GATE_PERCENT = 3.0
GATE_ORDER = 1.8

FAMILY_ORDER = {"dg_p1": 2.0, "dg_p2": 3.0, "dg_p3": 5.0, "fd2": 2.0, "fd4": 4.0}

#: Pairs chosen to cover every interesting regime of the order law.
PAIRS = (
    ("dg_p1", "fd2"),   # expected degenerate: SIPG p=1 on a Cartesian grid IS FD2
    ("dg_p2", "fd2"),   # min(3, 2) = 2
    ("dg_p3", "fd2"),   # min(5, 2) = 2, stronger pre-asymptotic buffering
    ("dg_p2", "fd4"),   # min(3, 4) = 3
    ("dg_p3", "fd4"),   # min(5, 4) = 4
)

#: Ladder used for the order law (all families are affordable here).
LADDER = (8, 16, 32, 64)

#: Published ladder of the anchor (report #19236, table "跨求解器差"), quoted for
#: the protocol's *external* consistency check.  These numbers are NOT
#: recomputed here -- the cavity solvers live in another repository -- so they
#: are labelled ``cited`` everywhere they appear.
ANCHOR_CITED = {
    "source": "tensor_reports#19236 (GAP-DIFFLBM-XSOLVER-ORDER ladder)",
    "status": "cited_not_recomputed",
    "resolutions": [65.0, 97.0, 129.0, 161.0],
    "percent_v_normalised": [3.4950, 1.6746, 0.9321, 0.5569],
    "percent_lid_speed_normalised": [0.8896, 0.4256, 0.2367, 0.1506],
    "family_orders": [2.0, 2.0],
}


# ---------------------------------------------------------------------------
# families
# ---------------------------------------------------------------------------
def exact_field(n: int) -> np.ndarray:
    """Exact solution sampled on the ``(n+1) x (n+1)`` node set ``x_i = i/n``."""
    x = np.linspace(0.0, 1.0, n + 1)
    return np.sin(np.pi * x)[:, None] * np.sin(np.pi * x)[None, :]


def _legendre_row(degree: int, xi: float) -> np.ndarray:
    return np.array([
        np.polynomial.legendre.legval(xi, np.eye(degree + 1)[k]) for k in range(degree + 1)
    ])


def dg_nodal_trace(coeff, n: int) -> np.ndarray:
    """Evaluate a SIPG solution at the shared nodes ``x_i = i/n``.

    Each node is an element face; the DG field is double valued there, so the
    trace is the mean of the two one-sided limits (a boundary node uses the one
    limit it has).  The resulting value is the quantity a downstream consumer of
    the DG field would see on that node set.
    """
    c = np.asarray(coeff.detach().cpu().numpy() if hasattr(coeff, "detach") else coeff)
    nx, ny, dx, _ = c.shape
    degree = dx - 1
    out = np.zeros((n + 1, n + 1))
    traces = {+1.0: _legendre_row(degree, 1.0), -1.0: _legendre_row(degree, -1.0)}
    for i in range(n + 1):
        if i == 0:
            cx = [(0, -1.0)]
        elif i == n:
            cx = [(n - 1, +1.0)]
        else:
            cx = [(i - 1, +1.0), (i, -1.0)]
        for j in range(n + 1):
            if j == 0:
                cy = [(0, -1.0)]
            elif j == n:
                cy = [(n - 1, +1.0)]
            else:
                cy = [(j - 1, +1.0), (j, -1.0)]
            total = 0.0
            count = 0
            for ei, xv in cx:
                for ej, yv in cy:
                    total += float(np.einsum("k,l,kl->", traces[xv], traces[yv], c[ei, ej]))
                    count += 1
            out[i, j] = total / count
    return out


def solve_dg(n: int, degree: int) -> np.ndarray:
    """SIPG DG solution sampled on the shared ``(n+1) x (n+1)`` node set."""
    _, _, coeff, _ = solve_poisson_2d(ncells=n, degree=degree, backend="sparse")
    return dg_nodal_trace(coeff, n)


def _manufactured_rhs(x: float, y: float) -> float:
    return 2.0 * math.pi ** 2 * math.sin(math.pi * x) * math.sin(math.pi * y)


def solve_fd2(n: int) -> np.ndarray:
    """5-point finite differences on the interior nodes (2nd order)."""
    h = 1.0 / n
    m = n - 1
    index = lambda i, j: i * m + j
    rows, cols, vals = [], [], []
    rhs = np.zeros(m * m)
    for i in range(m):
        for j in range(m):
            p = index(i, j)
            rows.append(p), cols.append(p), vals.append(4.0 / h ** 2)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < m and 0 <= jj < m:
                    rows.append(p), cols.append(index(ii, jj)), vals.append(-1.0 / h ** 2)
            rhs[p] = _manufactured_rhs((i + 1) * h, (j + 1) * h)
    matrix = sp.csr_matrix((vals, (rows, cols)), shape=(m * m, m * m))
    interior = spl.spsolve(matrix, rhs).reshape(m, m)
    out = np.zeros((n + 1, n + 1))
    out[1:-1, 1:-1] = interior
    return out


def solve_fd4(n: int) -> np.ndarray:
    """9-point compact (Mehrstellen) finite differences (4th order).

    ``L9 u = (1 - h^2/12 * Delta)(-F)`` with ``L9`` the ``(1,4,1)/(4,-20,4)``
    stencil scaled by ``1/(6h^2)``.  Because ``Delta sin sin = -2*pi^2 sin sin``
    the right-hand side is ``-F (1 - h^2 pi^2 / 6)``; using ``F`` unmodified
    leaves a 2nd-order defect, which is how the sign of this correction was
    pinned down here (the wrong sign gave an O(1) error, the missing one an
    O(h^2) rate).
    """
    h = 1.0 / n
    m = n - 1
    index = lambda i, j: i * m + j
    rows, cols, vals = [], [], []
    rhs = np.zeros(m * m)
    stencil = {(0, 0): -20.0, (1, 0): 4.0, (-1, 0): 4.0, (0, 1): 4.0, (0, -1): 4.0,
               (1, 1): 1.0, (1, -1): 1.0, (-1, 1): 1.0, (-1, -1): 1.0}
    for i in range(m):
        for j in range(m):
            p = index(i, j)
            for (di, dj), weight in stencil.items():
                ii, jj = i + di, j + dj
                if 0 <= ii < m and 0 <= jj < m:
                    rows.append(p), cols.append(index(ii, jj))
                    vals.append(weight / (6.0 * h ** 2))
            x, y = (i + 1) * h, (j + 1) * h
            rhs[p] = -_manufactured_rhs(x, y) * (1.0 - h ** 2 * math.pi ** 2 / 6.0)
    matrix = sp.csr_matrix((vals, (rows, cols)), shape=(m * m, m * m))
    interior = spl.spsolve(matrix, rhs).reshape(m, m)
    out = np.zeros((n + 1, n + 1))
    out[1:-1, 1:-1] = interior
    return out


FAMILIES = {
    "dg_p1": lambda n: solve_dg(n, 1),
    "dg_p2": lambda n: solve_dg(n, 2),
    "dg_p3": lambda n: solve_dg(n, 3),
    "fd2": solve_fd2,
    "fd4": solve_fd4,
}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(ladder=LADDER) -> dict:
    fields: dict[str, dict[int, np.ndarray]] = {name: {} for name in FAMILIES}
    own_error: dict[str, dict[int, float]] = {name: {} for name in FAMILIES}
    for name, solver in FAMILIES.items():
        for n in ladder:
            field = solver(n)
            fields[name][n] = field
            own_error[name][n] = max_point_error(field, exact_field(n))

    pairs = []
    for a, b in PAIRS:
        diffs = [cross_family_difference(fields[a][n], fields[b][n]) for n in ladder]
        verdict = cross_family_verdict(
            ladder, diffs,
            gate_percent=GATE_PERCENT,
            gate_order=GATE_ORDER,
            family_orders=[FAMILY_ORDER[a], FAMILY_ORDER[b]],
            family_errors_a=[own_error[a][n] for n in ladder],
            family_errors_b=[own_error[b][n] for n in ladder],
            scale=1.0,          # field amplitude of the manufactured solution
            field_norm=1.0,
        )
        # triangle-inequality check: |A-B| <= e_A + e_B, and how tight it is
        tightness = [
            diffs[k] / (own_error[a][n] + own_error[b][n])
            for k, n in enumerate(ladder)
        ]
        pairs.append({
            "a": a, "b": b,
            "family_orders": [FAMILY_ORDER[a], FAMILY_ORDER[b]],
            "order_bracket": list(verdict.order_bracket),
            "differences": diffs,
            "percent": [percent_of(d, 1.0) for d in diffs],
            "triangle_ratio": tightness,
            "verdict": verdict.as_dict(),
            "required_resolution_for_gate": required_resolution(ladder, diffs, GATE_PERCENT / 100.0),
        })

    anchor_pct = ANCHOR_CITED["percent_v_normalised"]
    anchor_diffs = [p / 100.0 for p in anchor_pct]
    anchor_verdict = cross_family_verdict(
        ANCHOR_CITED["resolutions"], anchor_diffs,
        gate_percent=GATE_PERCENT, gate_order=GATE_ORDER,
        family_orders=ANCHOR_CITED["family_orders"], scale=1.0, field_norm=1.0,
        # No per-family own error is published for the cavity ladder, so the
        # alias detector cannot run: the ladder can show agreement, never
        # accuracy.  That absence is itself the finding and is reported as such.
    )
    # Reconciliation factor between the two normalisations of the *same*
    # absolute difference: pct_v / pct_lid = U_lid / max|v|.  Both percentages
    # come from the cited ladder, so the ratio is derived, not assumed.
    ratio = ANCHOR_CITED["percent_v_normalised"][0] / ANCHOR_CITED["percent_lid_speed_normalised"][0]
    anchor = {
        "source": ANCHOR_CITED["source"],
        "status": ANCHOR_CITED["status"],
        "resolutions": ANCHOR_CITED["resolutions"],
        "percent_v_normalised": anchor_pct,
        "percent_lid_normalised": ANCHOR_CITED["percent_lid_speed_normalised"],
        "normalisation_ratio": ratio,
        "verdict": anchor_verdict.as_dict(),
        "required_resolution_for_gate": required_resolution(
            ANCHOR_CITED["resolutions"], anchor_diffs, GATE_PERCENT / 100.0
        ),
        "alias_detector": "not_run: per-family own errors are absent from the cited ladder",
        "asymptotic_subset_percent": anchor_pct[1:],
        "asymptotic_subset_all_below_gate": all(p < GATE_PERCENT for p in anchor_pct[1:]),
    }

    return {
        "problem": "-Delta u = 2*pi^2 sin(pi x) sin(pi y) on (0,1)^2, u=0 on boundary",
        "ladder": list(ladder),
        "gate_percent": GATE_PERCENT,
        "gate_order": GATE_ORDER,
        "half_gate_budget_percent": half_gate_budget(GATE_PERCENT),
        "family_orders_theoretical": FAMILY_ORDER,
        "own_max_point_error": {k: {str(n): v for n, v in d.items()} for k, d in own_error.items()},
        "own_error_observed_order": {
            k: [float(observed_order(ladder, [d[n] for n in ladder]))]
            for k, d in own_error.items()
        },
        "own_error_pairwise_order": {
            k: pairwise_orders(ladder, [d[n] for n in ladder]) for k, d in own_error.items()
        },
        "pairs": pairs,
        "anchor_cited": anchor,
    }


def table_markdown(result: dict) -> str:
    lines = [
        "| pair | p bracket | grids | diff % (finest->coarsest) | p_finest | p_lsq | attribution | gate<3% | gate p>=1.8 | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for pair in result["pairs"]:
        v = pair["verdict"]
        pct = " / ".join(f"{x:.3g}" for x in pair["percent"][::-1])
        lines.append(
            f"| {pair['a']} vs {pair['b']} | {pair['order_bracket'][0]:.0f}-{pair['order_bracket'][1]:.0f} | "
            f"{'-'.join(str(int(n)) for n in result['ladder'])} | {pct} | "
            f"{v['p_finest']:.2f} | {v['p_lsq']:.2f} | {v['order_attribution']} | "
            f"{v['gate_pct_passed']} | {v['gate_order_passed']} | {'PASS' if v['passed'] else 'not admissible/FAIL'} |"
        )
    return "\n".join(lines)


def figure(result: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    n = np.array(result["ladder"], dtype=float)
    for pair in result["pairs"]:
        d = np.array(pair["differences"], dtype=float)
        ax.loglog(n, 100 * d, "o-", label=f"{pair['a']} vs {pair['b']} (p bracket {pair['order_bracket'][0]:.0f}-{pair['order_bracket'][1]:.0f})")
    anchor_n = np.array(result["anchor_cited"]["resolutions"])
    ax.loglog(anchor_n, result["anchor_cited"]["percent_v_normalised"], "s--",
              color="k", label="anchor cavity ladder (cited #19236)")
    for order, style in ((2, ":"), (4, "--")):
        ref = (n / n[0]) ** (-order)
        ax.loglog(n, 100 * result["pairs"][1]["differences"][0] * ref, style, color="grey",
                  label=f"slope reference p={order}")
    ax.axhline(result["gate_percent"], color="tab:red", lw=1, label="3% gate")
    ax.set_xlabel("resolution n (same node set for both families)")
    ax.set_ylabel("cross-family difference [% of field amplitude]")
    ax.set_title("cross-family difference decays at min(p_A, p_B)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    names = list(result["family_orders_theoretical"])
    own = [result["own_error_observed_order"][k][0] for k in names]
    theory = [result["family_orders_theoretical"][k] for k in names]
    idx = np.arange(len(names))
    ax.bar(idx - 0.2, theory, 0.4, label="theoretical order")
    ax.bar(idx + 0.2, own, 0.4, label="observed (own error)")
    ax.set_xticks(idx)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("observed order")
    ax.set_title("family self-convergence on the shared ladder")
    ax.axhline(result["gate_order"], color="tab:red", lw=1, ls=":", label="gate p=1.8")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="coarse-ladder smoke run (no figure)")
    args = parser.parse_args()
    ladder = LADDER[:-1] if args.quick else LADDER
    result = run(ladder)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    (OUT / "table.md").write_text(table_markdown(result) + "\n")
    if not args.quick:
        figure(result, OUT / "figures" / "xfamily_order.png")
    print(table_markdown(result))
    print(f"\nanchor ladder (cited): worst pct {max(result['anchor_cited']['percent_v_normalised']):.3f}%"
          f" -> p_finest {result['anchor_cited']['verdict']['p_finest']:.2f},"
          f" required n for gate {result['anchor_cited']['required_resolution_for_gate']:.1f}")
    print(f"artifacts: {OUT}")


if __name__ == "__main__":
    main()