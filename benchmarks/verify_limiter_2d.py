"""Verification of the two-dimensional shock limiters.

Probes (all numbers in the JSON are produced here, nothing transcribed):

A. ``detector``       -- troubled-cell flags on the squeezed-Sod averages,
                         legacy wrapped neighbour vs periodicity-aware one.
B. ``reconstruction`` -- sub-cell profile error of the 2-D TVB/WENO limiters on
                         a smooth analytic field, with the wrapped (legacy) and
                         the periodicity-aware troubled-cell test.
C. ``sod``            -- squeezed Sod tube against the exact Riemann solution
                         for ``limiter in (None, "tvb", "weno")`` (3 % gate)
                         plus overshoot / total-variation statistics.
D. ``conservation``   -- periodic smooth flow: total mass/momentum/energy drift
                         while the limiters are active.

Usage::

    PYTHONPATH=. python -m benchmarks.verify_limiter_2d --threads 4
    PYTHONPATH=. python -m benchmarks.verify_limiter_2d --skip-sod   # fast path
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks.sod_exact import cell_averages as sod_cell_averages
from tensordg import euler_dg_2d as dg2
from tensordg.euler_dg_2d import cell_fields

HERE = Path(__file__).resolve().parent.parent
OUTPUT = HERE / "validation" / "limiter_2d"


def relative_l2(numerical: np.ndarray, exact: np.ndarray) -> float:
    return float(np.sqrt(np.mean((numerical - exact) ** 2))) / float(
        np.sqrt(np.mean(exact**2))
    )


def detection_probe(ncells=100):
    """Flags for a constant boundary cell that sits next to a real jump."""
    x = (np.arange(ncells) + 0.5) / ncells
    averages = torch.zeros(ncells, 1, 4, dtype=torch.float64)
    averages[..., 0] = torch.as_tensor(np.where(x < 0.5, 1.0, 0.125), dtype=torch.float64)[:, None]
    averages[..., 3] = torch.as_tensor(np.where(x < 0.5, 1.0, 0.1), dtype=torch.float64)[:, None]
    wrapped, _ = dg2._troubled_1d(averages, 0, 1.0 / ncells, 1.0, True)
    clamped, _ = dg2._troubled_1d(averages, 0, 1.0 / ncells, 1.0, False)
    return {
        "case": "squeezed Sod initial averages (cell 0 is uniform rho = 1)",
        "flagged_wrapped": [int(i) for i in torch.nonzero(wrapped[:, 0]).flatten()],
        "flagged_periodic_aware": [int(i) for i in torch.nonzero(clamped[:, 0]).flatten()],
        "note": "the wrapped test invents a neighbour with rho = 0.125 for cell 0",
    }


def reconstruction_probe(ncells=16, degree=2):
    """Sub-cell error of the limiters on the smooth analytic field cos(pi x)."""
    dx, dy = 1.0 / ncells, 0.25
    x = (torch.arange(ncells, dtype=torch.float64) + 0.5) * dx
    y = torch.tensor([0.125], dtype=torch.float64)
    order = max(degree + 2, 3)
    nodes, weights, basis, _, _ = dg2.legendre_operators(degree, order)

    def initial(xx, yy):
        shape = torch.broadcast_shapes(xx.shape, yy.shape)
        rho = torch.broadcast_to(1.0 + 0.2 * torch.cos(math.pi * xx), shape)
        return rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.ones_like(rho)

    coeff = dg2._initial_coefficients(initial, x, y, dx, dy, nodes, weights, basis, "cpu", torch.float64)
    physical = x[:, None] + 0.5 * dx * nodes[None, :]
    exact = 1.0 + 0.2 * torch.cos(math.pi * physical)
    scale = float(exact.abs().max())

    rows = []
    for name, limiter in (("tvb", dg2.apply_tvb_limiter_2d), ("weno", dg2.apply_weno_limiter_2d)):
        for legacy, flags in ((True, (True, True)), (False, (False, True))):
            limited = limiter(coeff.clone(), dx, dy, 1.0, flags)
            values = torch.einsum("xyklv,ak,bl->xyabv", limited, basis, basis)[..., 0, :, :, 0]
            error = (values[:, 0] - exact).abs()
            rows.append(
                {
                    "limiter": name,
                    "legacy_detector": legacy,
                    "periodic": list(flags),
                    "subcell_relative_l2": float(torch.sqrt((error**2).mean())) / scale,
                    "boundary_cell_relative_error": float(
                        torch.maximum(error[0].max(), error[-1].max())
                    )
                    / scale,
                    "interior_cell_relative_error": float(error[1:-1].max()) / scale,
                    "cell_average_preserved": bool(
                        torch.equal(limited[..., 0, 0, :], coeff[..., 0, 0, :])
                    ),
                }
            )
    return rows


def sod_probe(ncells=100, degree=1, legacy_tvb=False):
    _, reference = sod_cell_averages(ncells, 0.2)
    rows = []
    for limiter, legacy in ((None, False), ("tvb", legacy_tvb), ("tvb", False), ("weno", False)):
        started = time.time()
        if legacy:
            original = dg2._troubled_1d
            dg2._troubled_1d = lambda a, ax, sp, tv, periodic=True: original(a, ax, sp, tv, True)
            try:
                _, _, coeff, metadata = dg2.solve_euler_dg_2d_sod(
                    ncells=ncells, degree=degree, final_time=0.2, cfl=0.2, limiter=limiter
                )
            finally:
                dg2._troubled_1d = original
        else:
            _, _, coeff, metadata = dg2.solve_euler_dg_2d_sod(
                ncells=ncells, degree=degree, final_time=0.2, cfl=0.2, limiter=limiter
            )
        fields = cell_fields(coeff)
        row = {
            "limiter": limiter if limiter else "none (positivity only)",
            "legacy_detector": legacy,
            "steps": metadata["steps"],
            "wall_seconds": time.time() - started,
        }
        for name in ("density", "pressure"):
            numerical = fields[name][:, 0].detach().cpu().numpy()
            exact = np.asarray(reference[name], dtype=float)
            row[f"{name}_relative_l2"] = relative_l2(numerical, exact)
        density = fields["density"][:, 0].detach().cpu().numpy()
        row.update(
            density_min=float(density.min()),
            density_max=float(density.max()),
            density_total_variation=float(np.abs(np.diff(density)).sum()),
            density_overshoot=float(density.max() - 1.0),
            density_profile=density.tolist(),
            density_exact=np.asarray(reference["density"], dtype=float).tolist(),
        )
        rows.append(row)
        print(
            f"  sod {row['limiter']:>22s} legacy={int(legacy)}: "
            f"rho {row['density_relative_l2'] * 100:.3f}%  "
            f"p {row['pressure_relative_l2'] * 100:.3f}%  "
            f"rho_max {row['density_max']:.4f}  TV {row['density_total_variation']:.4f}"
        )
    return rows


def conservation_probe(ncells=16, degree=2, final_time=0.2):
    outputs = {}
    dx = 10.0 / ncells
    for limiter in ("tvb", "weno"):
        def initial(x, y):
            return dg2.vortex_primitive(x, y, 0.0, 5.0, (1.0, 1.0))

        kwargs = dict(
            ncells=(ncells, ncells), degree=degree, domain=((-5.0, 5.0), (-5.0, 5.0)),
            cfl=0.2, periodic=(True, True),
        )
        _, _, reference, _ = dg2.solve_euler_dg_2d(initial, final_time=0.0, limiter=None, **kwargs)
        _, _, coeff, metadata = dg2.solve_euler_dg_2d(
            initial, final_time=final_time, limiter=limiter, **kwargs
        )
        start = reference[..., 0, 0, :].sum(dim=(0, 1)) * dx * dx
        end = coeff[..., 0, 0, :].sum(dim=(0, 1)) * dx * dx
        drift = ((end - start).abs() / start.abs()).tolist()
        outputs[limiter] = {
            "steps": metadata["steps"],
            "relative_drift_mass_momentum_x_momentum_y_energy": drift,
            "max_relative_drift": float(max(drift)),
        }
        print(f"  conservation {limiter}: max drift {max(drift):.2e}")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip-sod", action="store_true")
    parser.add_argument("--skip-conservation", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT / "verify.json"))
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.threads))

    started = time.time()
    report = {"environment": {"torch": torch.__version__, "threads": args.threads}}

    print("A. troubled-cell detector")
    report["detector"] = detection_probe()
    print("   wrapped:", report["detector"]["flagged_wrapped"],
          " periodic-aware:", report["detector"]["flagged_periodic_aware"])

    print("B. sub-cell reconstruction of a smooth analytic field")
    report["reconstruction"] = reconstruction_probe()
    for row in report["reconstruction"]:
        print(
            f"   {row['limiter']} legacy={int(row['legacy_detector'])}: "
            f"sub-cell L2 {row['subcell_relative_l2']:.3e}  "
            f"boundary {row['boundary_cell_relative_error']:.3e}  "
            f"interior {row['interior_cell_relative_error']:.3e}"
        )

    if not args.skip_sod:
        print("C. squeezed Sod vs exact Riemann solution (P1, 100 cells)")
        report["sod"] = sod_probe()
    if not args.skip_conservation:
        print("D. periodic conservation")
        report["conservation"] = conservation_probe()

    report["wall_seconds"] = time.time() - started
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"wrote {output} ({report['wall_seconds']:.1f}s)")


if __name__ == "__main__":
    main()