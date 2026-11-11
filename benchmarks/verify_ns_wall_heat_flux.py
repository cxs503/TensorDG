"""Wall heat-flux order study for the isothermal channel of ``tensordg.ns_dg_2d``.

Context
-------
``tests/test_ns_heat_conduction.py`` carries two ``xfail(strict=True)`` pins that
record an O(1) energy residual of the *exact* steady conduction state on the
wall row, and its documentation attributes that to an *inconsistent* isothermal
wall energy closure.  This benchmark measures the three quantities that decide
the question, and nothing else:

* ``wall_flux``      : the total energy flux the wall closure puts into the
  strong-form divergence, compared with the exact ``-kappa dT/dn = kappa/2``;
* ``interior_flux``  : the same quantity on the neighbouring interior face;
* ``row_residual``   : the wall-row energy residual (the pinned quantity) and
  the volume-weighted global residual ``||R||`` of the whole discrete operator.

If the wall flux converges to the exact value, the closure is *consistent* and
the stagnation of ``row_residual`` is a measure-``O(h)`` boundary effect whose
strong-form residual is ``O(h) / h`` -- not a solution-accuracy defect.  If it
does not converge, the closure itself is wrong.

The analytic state is ``u = v = 0``, ``p = p0``, ``T = T_hot + (T_cold - T_hot) y``,
``rho = p0 / T`` (it satisfies the compressible equations exactly, and the energy
equation degenerates to ``kappa laplace(T) = 0``).

Run::

    PYTHONPATH=. python -m benchmarks.verify_ns_wall_heat_flux

writes ``validation/ns_heat_conduction/wall_flux_order.json``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

import tensordg.ns_dg_2d as ns

MU = 0.1
PRANDTL = 0.71
KAPPA = MU * ns.CP / PRANDTL
T_HOT, T_COLD, PRESSURE = 1.0, 0.5, 1.0
NX = 2
NY_SET = (8, 16, 32, 64)
OUT = Path("validation/ns_heat_conduction/wall_flux_order.json")


def _steady(x, y):
    ref = x * y + 0.0 * x + 0.0 * y
    temperature = T_HOT + (T_COLD - T_HOT) * (y + 0.0 * x)
    return (PRESSURE / temperature, torch.zeros_like(ref), torch.zeros_like(ref),
            PRESSURE * torch.ones_like(ref))


def _case(ny, degree, wall_gradient):
    dx, dy = 1.0 / NX, 1.0 / ny
    nodes, weights, basis, derivative, signs = ns.legendre_operators(
        degree, degree + 2, torch.device("cpu"), torch.float64)
    x = (torch.arange(NX, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    coeff = ns._initial_coefficients(_steady, x, y, dx, dy, nodes, weights, basis,
                                     torch.device("cpu"), torch.float64)
    penalty = 2.0 * (degree + 1.0) ** 2
    boundary = {"xmin": None, "xmax": None,
                "ymin": {"velocity": (0.0, 0.0), "temperature": T_HOT},
                "ymax": {"velocity": (0.0, 0.0), "temperature": T_COLD}}
    rhs = ns._rhs_ns(coeff, basis, derivative, weights, signs, dx, dy,
                     (True, False), boundary, MU, KAPPA, penalty, wall_gradient)
    right_y, left_y, _, _ = ns._axis_fluxes_ns(
        coeff, basis, derivative, signs, weights, dx, dy, False,
        boundary["ymin"], boundary["ymax"], MU, KAPPA, penalty, 1, wall_gradient)
    # right_y / left_y hold the *projected* face flux int F* phi_k dxi; the k = 0
    # mode is 2 F* for a face-constant flux, so divide by two quadrature weights.
    weight0 = 2.0
    wall_flux = float(left_y[:, 0, 0, 3].mean()) / weight0
    interior_flux = float(right_y[:, 0, 0, 3].mean()) / weight0
    # volume-weighted residual: mass-matrix norm of R over the domain
    modal = rhs[..., 3]
    modes = torch.arange(degree + 1, dtype=torch.float64)
    mass = 1.0 / ((2.0 * modes + 1.0) * (2.0 * modes + 1.0))
    l2 = float((modal.pow(2) * mass.view(1, 1, *mass.shape, 1)).sum().sqrt()) * math.sqrt(dx * dy / 4.0)
    return {
        "ny": ny,
        "degree": degree,
        "wall_gradient": wall_gradient,
        "wall_flux": wall_flux,
        "interior_flux": interior_flux,
        "row_residual": float(modal[:, 0].abs().max()),
        "row_residual_inner": float(modal[:, 1:-1].abs().max()),
        "l2_residual": l2,
    }


def observed_order(values, cells):
    xs = [math.log(n) for n in cells]
    ys = [math.log(v) for v in values]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return -sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def main():
    exact_flux = KAPPA * 0.5
    rows = []
    summary = {"kappa": KAPPA, "exact_wall_flux": exact_flux,
               "prandtl": PRANDTL, "mu": MU,
               "temperatures": {"hot": T_HOT, "cold": T_COLD}, "pressure": PRESSURE}
    for degree in (1, 2):
        for wall_gradient in ("taylor", "face"):
            cases = [_case(ny, degree, wall_gradient) for ny in NY_SET]
            for case in cases:
                case["wall_flux_error"] = case["wall_flux"] - exact_flux
                case["interior_flux_error"] = case["interior_flux"] - exact_flux
            orders = {
                "wall_flux_error": observed_order([abs(c["wall_flux_error"]) for c in cases], NY_SET),
                "interior_flux_error": observed_order(
                    [abs(c["interior_flux_error"]) for c in cases], NY_SET),
                "row_residual": observed_order([c["row_residual"] for c in cases], NY_SET),
                "row_residual_inner": observed_order(
                    [c["row_residual_inner"] for c in cases], NY_SET),
                "l2_residual": observed_order([c["l2_residual"] for c in cases], NY_SET),
            }
            summary.setdefault("orders", {})[f"p{degree}_{wall_gradient}"] = orders
            rows.extend(cases)
    summary["cases"] = rows
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))

    print(f"kappa = {KAPPA:.6f}   exact wall energy flux = {exact_flux:.6f}")
    header = (f"{'deg':>3} {'wg':>6} {'ny':>4} {'wall_flux':>10} {'err':>10}"
              f" {'interior':>10} {'err':>10} {'row_res':>9} {'l2_res':>10}")
    print(header)
    for case in rows:
        print(f"{case['degree']:>3} {case['wall_gradient']:>6} {case['ny']:>4}"
              f" {case['wall_flux']:10.6f} {case['wall_flux_error']:+10.2e}"
              f" {case['interior_flux']:10.6f} {case['interior_flux_error']:+10.2e}"
              f" {case['row_residual']:9.4f} {case['l2_residual']:10.3e}")
    print("\nobserved orders (log-log slope over "
          f"{list(NY_SET)}):")
    for key, orders in summary["orders"].items():
        print(f"  {key}: " + "  ".join(f"{name}={value:+.2f}" for name, value in orders.items()))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()