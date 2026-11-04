"""Is the 2-D limiter order-preserving on smooth flow?  (validation debt m3714)

Background
----------
Mission m2549 fixed two defects of ``tensordg.euler_dg_2d.apply_tvb_limiter_2d``:
the troubled-cell test was not scale consistent (raw one-sided differences
``O(dx u')`` were compared against ``M dx**2``), and for ``degree >= 2`` the
mask that clears the higher modes was broadcast on the wrong axes.  The fix was
verified in 1-D only; ``docs/limiter_policy_verification.md`` §5 item 2 records
the remaining gap verbatim:

    「二维限制器（tensordg/euler_dg_2d.py 的可选 TVB）沿用同一个
     _troubled_cell_indicator，本次修复对它连带生效但未单独验证
     （未做二维光滑阶数测试）」

That is the debt this module pays.  The isentropic vortex is an *exact*
solution of the 2-D Euler equations on a periodic box, so the measured error is
against analysis, not against another numerical solution.

What is measured
----------------
For the P2 ladder n = 8, 16, 32 (and P1 as a cheap control) on the analytic
vortex at ``T = 1``:

* ``none``            -- no limiter (positivity fallback only): the reference
                         design order is ``p + 1``;
* ``tvb`` with ``M = 0``   -- strict TVD test, the *negative control*: it must
                             clip the smooth extremum of the vortex and lose
                             order, otherwise the test has no discriminating
                             power;
* ``tvb`` with ``M = 1, 10, 100`` -- the scale-consistent tolerance is
                             ``M dx**2``, so for a fixed solution there is a
                             mesh-independent ``M`` above which the limiter is
                             *inert* (``M >= M_threshold``) and the unlimited
                             design order must be reproduced bit for bit;
* ``weno`` with the same ``M`` -- reported for completeness; a WENO
                         reconstruction of a smooth solution is not expected to
                         keep the modal design order.

``M_threshold`` is estimated independently from the data: the limiter's measure
is ``|d_left - minmod(d_left, d_right)| = O(dx**2 u'')``, so dividing the
largest second difference of the *exact* cell averages by ``dx**2`` gives the
``M`` at which the test can no longer fire.  That number is computed here and
compared with the ``M`` at which the measured error becomes identical to the
unlimited run -- a genuine prediction, not a fit.

Run with::

    PYTHONPATH=. python -m benchmarks.limiter_2d_order_study

Evidence lands in ``validation/limiter_2d_order/`` (JSON) and the figures in
``docs/figures/limiter_2d_order/``.  Cost: seconds on one core at these meshes
(the whole ladder is far below the 128-core node budget), see the capacity note
in the verification document.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

import tensordg.euler_dg_2d as dg2
from tensordg.euler_2d import vortex_primitive

from .verify_euler_dg_2d import _vortex_error

DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))
FINAL_TIME = 1.0
STRENGTH = 5.0
BACKGROUND = (1.0, 1.0)

OUT_DIR = Path("validation/limiter_2d_order")
FIG_DIR = Path("docs/figures/limiter_2d_order")

# (label, limiter, tvb_constant)
CONFIGS = (
    ("none", None, 1.0),
    ("tvb_M0", "tvb", 0.0),
    ("tvb_M1", "tvb", 1.0),
    ("tvb_M4", "tvb", 4.0),
    ("tvb_M10", "tvb", 10.0),
    ("tvb_M100", "tvb", 100.0),
    ("weno_M0", "weno", 0.0),
    ("weno_M10", "weno", 10.0),
)


def _run(ncells, degree, limiter, tvb_constant):
    nodes, _, basis, _, _ = dg2.legendre_operators(degree, max(degree + 2, 3))
    _, _, coeff, metadata = dg2.solve_euler_dg_2d_vortex(
        ncells=ncells,
        degree=degree,
        final_time=FINAL_TIME,
        strength=STRENGTH,
        background_velocity=BACKGROUND,
        limiter=limiter,
        tvb_constant=tvb_constant,
        domain=DOMAIN,
    )
    errors = _vortex_error(coeff, basis, nodes, DOMAIN, FINAL_TIME,
                           strength=STRENGTH, background=BACKGROUND)
    return errors, metadata, coeff


def _second_difference_scale(degree, ncells):
    """Mesh-independent estimate of ``M`` above which the TVB test cannot fire.

    The limiter compares ``|d_left - minmod(d_left, d_right)|`` against
    ``M dx**2``.  For the exact solution the largest such deviation is bounded
    by the largest second difference of the cell averages, so
    ``M_threshold = max |d_left - minmod(d_left, d_right)| / dx**2`` evaluated on
    the exact vortex at the finest mesh in the ladder.  A conservative extra
    factor is *not* applied: the point of the exercise is to test whether the
    measured inertness threshold matches this prediction.
    """
    (xmin, xmax), (ymin, ymax) = DOMAIN
    dx = (xmax - xmin) / ncells
    dy = (ymax - ymin) / ncells
    quad = max(degree + 2, 3)
    node_np = torch.linspace(-1.0, 1.0, 64, dtype=torch.float64)
    x = xmin + (torch.arange(ncells, dtype=torch.float64) + 0.5) * dx
    y = ymin + (torch.arange(ncells, dtype=torch.float64) + 0.5) * dy
    # Exact cell averages of the conserved variables via tensor Gauss averaging.
    gauss_x, weight_x = torch.as_tensor(_leggauss(quad)[0]), torch.as_tensor(_leggauss(quad)[1])
    xq = x[:, None, None, None] + 0.5 * dx * gauss_x[None, None, :, None]
    yq = y[None, :, None, None] + 0.5 * dy * gauss_x[None, None, None, :]
    prim = vortex_primitive(xq, yq, FINAL_TIME, STRENGTH, BACKGROUND)
    rho, u, v, p = prim
    conserved = _conservative_quad(rho, u, v, p)
    weight = weight_x[:, None] * weight_x[None, :]
    averages = 0.25 * torch.einsum("xyabv,ab->xyv", conserved, weight)

    worst = 0.0
    per_axis = {}
    for axis, spacing in ((0, dx), (1, dy)):
        plus = torch.roll(averages, -1, dims=axis)
        minus = torch.roll(averages, 1, dims=axis)
        d_left = averages - minus
        d_right = plus - averages
        mm = torch.zeros_like(d_left)
        same = d_left * d_right > 0.0
        mm = torch.where(same, torch.sign(d_left) * torch.minimum(d_left.abs(), d_right.abs()), mm)
        dev = (d_left - mm).abs().max().item()
        per_axis["xy"[axis]] = dev / spacing ** 2
        worst = max(worst, dev / spacing ** 2)
    return {"M_threshold_estimate": worst, "per_axis": per_axis,
            "ncells_used": ncells, "_unused_node": float(node_np[0])}


def _leggauss(n):
    import numpy as np

    return np.polynomial.legendre.leggauss(n)


def _conservative_quad(rho, u, v, p):
    gamma = dg2.GAMMA
    energy = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)
    return torch.stack([rho, rho * u, rho * v, energy], dim=-1)


def _eoc(coarse, fine):
    if coarse <= 0.0 or fine <= 0.0:
        return float("nan")
    return math.log(coarse / fine) / math.log(2.0)


def _configs_by_label(labels):
    by_label = {label: (label, limiter, tvb) for label, limiter, tvb in CONFIGS}
    return tuple(by_label[label] for label in labels)


def run_ladder(degree, meshes, configs=CONFIGS):
    table = {}
    coefficients = {}
    for label, limiter, tvb in configs:
        row = {}
        for n in meshes:
            errors, metadata, coeff = _run(n, degree, limiter, tvb)
            row[n] = {
                "relative_l2": {k: v for k, v in errors.items()
                                if k != "pressure_max"},
                "steps": metadata["steps"],
            }
            coefficients[(label, n)] = coeff
        # observed order on the finest interval, per field
        n1, n2 = meshes[-2], meshes[-1]
        row["eoc"] = {
            f: _eoc(row[n1]["relative_l2"][f], row[n2]["relative_l2"][f])
            for f in ("density", "velocity_x", "velocity_y", "pressure")
        }
        row["limiter"] = limiter
        row["tvb_constant"] = tvb
        table[label] = row
    return table, coefficients


def _bit_identical(a, b):
    return bool(torch.equal(a, b))


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--degree", type=int, default=None,
                        help="restrict to one degree (default: 1 and 2)")
    parser.add_argument("--configs", default=None,
                        help="comma separated config labels (sharding)")
    parser.add_argument("--meshes", default="8,16,32")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", default=str(OUT_DIR / "study.json"))
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    meshes = tuple(int(v) for v in args.meshes.split(","))
    configs = _configs_by_label(args.configs.split(",")) if args.configs else CONFIGS
    degrees = (args.degree,) if args.degree is not None else (1, 2)
    results = {"final_time": FINAL_TIME, "strength": STRENGTH,
               "domain": DOMAIN, "background": BACKGROUND, "degrees": {}}

    for degree in degrees:
        table, coeffs = run_ladder(degree, meshes, configs)
        have_unlimited = ("none", meshes[-1]) in coeffs
        inert_by_config = {
            label: _bit_identical(coeffs[("none", meshes[-1])], coeffs[(label, meshes[-1])])
            for label, _, _ in configs
            if have_unlimited and (label, meshes[-1]) in coeffs
        }
        results["degrees"][str(degree)] = {
            "design_order": degree + 1,
            "meshes": list(meshes),
            "configs": table,
            "bit_identical_to_unlimited_on_finest_mesh": inert_by_config,
        }
        print(f"--- degree {degree} (design order {degree + 1}) ---")
        for label, _, _ in configs:
            row = table[label]
            worst = max(row[n]["relative_l2"][f] for n in meshes for f in
                        ("density", "velocity_x", "velocity_y", "pressure"))
            print(f"{label:9s} eoc_rho={row['eoc']['density']:.3f} "
                  f"eoc_p={row['eoc']['pressure']:.3f} worst_rel_l2={worst:.3e} "
                  f"inert={inert_by_config.get(label, 'n/a')}")

    results["M_threshold_estimate"] = _second_difference_scale(2, meshes[-1])
    print("M threshold estimate:", results["M_threshold_estimate"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print("wrote", out_path)
    return results


if __name__ == "__main__":
    main()