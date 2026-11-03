"""Verification suite for the high-order 2-D Euler DG module (mission m2400).

Run with ``PYTHONPATH=. python -m benchmarks.verify_euler_dg_2d``.

Four checks, each with a machine-decidable criterion:

A  cross-code residual identity -- the 2-D residual of ``y``-invariant data must
   equal the residual of the independently written 1-D DG solver
   (``tensordg/euler_dg_1d.py``) for every cell, mode and component.  This is
   the check the m2143 defect broke.
B  Sod tube extruded to 2-D vs the exact Riemann solution (3 % gate) for P1/P2.
C  smooth isentropic vortex -- observed convergence order vs the design order
   ``degree + 1`` and L2 error vs the 3 % gate.
D  four-quadrant Riemann problem (Lax & Liu 1998, config 3) -- positivity on all
   quadrature points plus mesh self-convergence (no closed-form solution).

Everything printed is written to ``validation/euler_dg_2d/`` as JSON/CSV so the
report numbers can be injected programmatically (no hand-copied digits).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import tensordg.euler_dg_2d as dg2
from tensordg.euler_1d import GAMMA
from tensordg.euler_2d import vortex_primitive

from benchmarks.sod_exact import cell_averages as sod_cell_averages
from benchmarks.sod_exact import sample as sod_sample

OUTPUT = Path("validation/euler_dg_2d")
LEFT_STATE = (1.0, 0.0, 1.0)
RIGHT_STATE = (0.125, 0.0, 0.1)
THREE_PERCENT = 0.03


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _reconstruct(coeff, basis):
    """Modal coefficients -> conservative state on the volume quadrature grid."""
    return torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)


def _primitive(state):
    return dg2._primitive(state)


def _vortex_error(coeff, basis, nodes, domain, final_time, strength=5.0,
                  background=(1.0, 1.0)):
    """Relative L2 error of (rho, u, v, p) against the exact vortex solution."""
    nx, ny = coeff.shape[0], coeff.shape[1]
    (xmin, xmax), (ymin, ymax) = domain
    dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
    x = xmin + (torch.arange(nx, dtype=coeff.dtype) + 0.5) * dx
    y = ymin + (torch.arange(ny, dtype=coeff.dtype) + 0.5) * dy
    xq = x[:, None, None, None] + 0.5 * dx * nodes[None, None, :, None]
    yq = y[None, :, None, None] + 0.5 * dy * nodes[None, None, None, :]
    exact = vortex_primitive(xq, yq, final_time, strength, background)
    state = _reconstruct(coeff, basis)
    rho, u, v, p = _primitive(state)
    weights = torch.as_tensor(
        np.polynomial.legendre.leggauss(nodes.numel())[1], dtype=coeff.dtype
    )
    weight = weights[:, None] * weights[None, :]
    errors = {}
    for name, numerical, reference in (
        ("density", rho, exact[0]),
        ("velocity_x", u, exact[1]),
        ("velocity_y", v, exact[2]),
        ("pressure", p, exact[3]),
    ):
        difference = (numerical - reference) ** 2
        denominator = reference ** 2
        norm = torch.sqrt(
            (difference * weight).sum() / (denominator * weight).sum()
        )
        errors[name] = float(norm)
    errors["pressure_max"] = float(errors["pressure"])
    return errors


def _extruded_state(ncells, degree, device, dtype):
    """y-invariant Sod state as **modal** coefficients (all x modes populated).

    Higher x modes are filled by an L2 projection of a smooth tanh profile so
    the check exercises the volume term as well, not only the P0 mode.
    """
    nodes, weights, basis, _, _ = dg2.legendre_operators(degree, max(degree + 2, 3))
    dx = 1.0 / ncells
    centres = (torch.arange(ncells, dtype=dtype) + 0.5) * dx
    xq = centres[:, None] + 0.5 * dx * nodes[None, :]

    sharpness = 220.0
    blend = 0.5 * (1.0 - torch.tanh(sharpness * (xq - 0.5)))
    rho = RIGHT_STATE[0] + (LEFT_STATE[0] - RIGHT_STATE[0]) * blend
    velocity = torch.zeros_like(rho)
    pressure = RIGHT_STATE[2] + (LEFT_STATE[2] - RIGHT_STATE[2]) * blend
    energy = pressure / (GAMMA - 1.0) + 0.5 * rho * velocity ** 2
    state = torch.stack((rho, velocity, energy), dim=-1)  # [c, q, 3]

    projected = torch.einsum("cqv,q,qk->ckv", state, weights, basis)
    modes = torch.arange(degree + 1, dtype=dtype)
    coeff_1d = (2.0 * modes + 1.0).view(1, -1, 1) / 2.0 * projected
    coeff = torch.zeros(ncells, 1, degree + 1, degree + 1, 4, dtype=dtype)
    coeff[:, 0, :, 0, 0] = coeff_1d[..., 0]
    coeff[:, 0, :, 0, 1] = coeff_1d[..., 1]
    coeff[:, 0, :, 0, 3] = coeff_1d[..., 2]
    return coeff, coeff_1d


# ---------------------------------------------------------------------------
# A. cross-code residual identity
# ---------------------------------------------------------------------------

def residual_identity(degrees=(1, 2), ncells=64):
    """2-D vs 1-D residual of y-invariant data (m2143 defect regression)."""
    from tensordg.euler_dg_1d import _rhs as rhs_1d

    results = {}
    for degree in degrees:
        _, weights, basis, derivative, signs = dg2.legendre_operators(
            degree, max(degree + 2, 3)
        )
        coeff, coeff_1d = _extruded_state(ncells, degree, "cpu", torch.float64)
        dx, dy = 1.0 / ncells, 0.25
        residual_2d = dg2._rhs(
            coeff, basis, derivative, weights, signs, dx, dy, (False, True)
        )
        residual_1d = rhs_1d(
            coeff_1d, dx, weights, basis, derivative, signs, torch.ones_like(signs)
        )
        # 1-D components (rho, rho u, E) map onto 2-D components (0, 1, 3).
        # every x mode (index 2), the single y mode l=0 (index 3), components 4.
        differences = {
            "density": (residual_2d[:, 0, :, 0, 0] - residual_1d[:, :, 0]).abs().max(),
            "momentum_x": (residual_2d[:, 0, :, 0, 1] - residual_1d[:, :, 1]).abs().max(),
            "energy": (residual_2d[:, 0, :, 0, 3] - residual_1d[:, :, 2]).abs().max(),
            "momentum_y": residual_2d[:, 0, :, 0, 2].abs().max(),
        }
        scale = max(float(residual_1d.abs().max()), 1e-300)
        results[str(degree)] = {
            "degree": degree,
            "cells": ncells,
            "max_abs_difference": {key: float(value) for key, value in differences.items()},
            "reference_scale": scale,
            "relative_difference": float(
                max(float(value) for value in differences.values()) / scale
            ),
            "one_d_residual_max": scale,
            "two_d_residual_max": float(residual_2d[:, 0, :, 0, :].abs().max()),
        }
    return results


# ---------------------------------------------------------------------------
# B. extruded Sod vs the exact Riemann solution
# ---------------------------------------------------------------------------

def extruded_sod(degrees=(1, 2), ncells=200, final_time=0.2, cfl=0.2):
    results = {}
    x_reference, reference = sod_cell_averages(ncells, final_time)
    x_reference = np.asarray(x_reference)
    for degree in degrees:
        started = time.time()
        x, _, coeff, metadata = dg2.solve_euler_dg_2d_sod(
            ncells=ncells, degree=degree, final_time=final_time, cfl=cfl
        )
        rho, u, _, pressure = dg2.cell_fields(coeff)
        density = rho[:, 0].detach().cpu().numpy()
        velocity = u[:, 0].detach().cpu().numpy()
        pressure_np = pressure[:, 0].detach().cpu().numpy()
        errors = {}
        for name, numerical in (
            ("density", density),
            ("velocity", velocity),
            ("pressure", pressure_np),
        ):
            absolute = float(np.sqrt(np.mean((numerical - reference[name]) ** 2)))
            norm = float(np.sqrt(np.mean(reference[name] ** 2)))
            errors[name] = {
                "relative_l2": absolute / norm,
                "relative_l2_percent": 100.0 * absolute / norm,
            }
        results[str(degree)] = {
            "degree": degree,
            "cells": ncells,
            "steps": metadata["steps"],
            "dt": metadata["dt"],
            "initial_projection": metadata["initial_projection"],
            "errors": errors,
            "wall_seconds": time.time() - started,
            "profile": {
                "x": np.asarray(x).tolist(),
                "density": density.tolist(),
                "pressure": pressure_np.tolist(),
            },
        }
    results["reference"] = {
        "x": x_reference.tolist(),
        "density": np.asarray(reference["density"]).tolist(),
        "pressure": np.asarray(reference["pressure"]).tolist(),
        "final_time": final_time,
    }
    return results


# ---------------------------------------------------------------------------
# C. smooth isentropic vortex convergence
# ---------------------------------------------------------------------------

def vortex_convergence(degree=2, meshes=(16, 24, 32), final_time=1.0, cfl=0.2):
    nodes, weights, basis, _, _ = dg2.legendre_operators(degree, max(degree + 2, 3))
    rows = []
    for ncells in meshes:
        started = time.time()
        _, _, coeff, metadata = dg2.solve_euler_dg_2d_vortex(
            ncells=ncells, degree=degree, final_time=final_time, cfl=cfl
        )
        errors = _vortex_error(coeff, basis, nodes, ((-5.0, 5.0), (-5.0, 5.0)), final_time)
        rows.append(
            {
                "degree": degree,
                "cells": ncells,
                "h": 10.0 / ncells,
                "dofs": metadata["dofs"],
                "steps": metadata["steps"],
                "dt": metadata["dt"],
                "errors": errors,
                "wall_seconds": time.time() - started,
                "density": float(coeff[..., 0].abs().max()),
            }
        )
        print(
            f"  vortex P{degree} {ncells:3d}^2: rho rel-L2 {errors['density']:.3e} "
            f"p rel-L2 {errors['pressure']:.3e} ({rows[-1]['wall_seconds']:.1f}s)"
        )
    for previous, current in zip(rows, rows[1:]):
        ratio = previous["h"] / current["h"]
        for field in ("density", "pressure"):
            current.setdefault("eoc", {})[field] = float(
                math.log(previous["errors"][field] / current["errors"][field])
                / math.log(ratio)
            )
    return {"degree": degree, "final_time": final_time, "rows": rows}


# ---------------------------------------------------------------------------
# D. four-quadrant Riemann problem
# ---------------------------------------------------------------------------

def four_quadrant(ncells=48, degree=1, final_time=0.25, cfl=0.2, coarse=32):
    """Positivity plus mesh self-convergence (no exact solution exists)."""
    results = {}
    for mesh in (coarse, ncells):
        started = time.time()
        _, _, coeff, metadata = dg2.solve_euler_dg_2d_riemann(
            ncells=mesh, degree=degree, final_time=final_time, cfl=cfl, limiter="tvb"
        )
        nodes, _, basis, _, _ = dg2.legendre_operators(degree, max(degree + 2, 3))
        state = _reconstruct(coeff, basis)
        rho, _, _, pressure = _primitive(state)
        results[str(mesh)] = {
            "cells": mesh,
            "steps": metadata["steps"],
            "dt": metadata["dt"],
            "min_density_quadrature": float(rho.min()),
            "min_pressure_quadrature": float(pressure.min()),
            "max_density": float(rho.max()),
            "finite": bool(torch.isfinite(state).all()),
            "wall_seconds": time.time() - started,
            "cell_density": coeff[..., 0, 0, 0][..., None, None].squeeze().tolist(),
        }
    # mesh self-convergence of the cell averages, restricted to the coarse grid.
    fine = np.asarray(results[str(ncells)]["cell_density"])
    block = ncells // coarse
    coarse_from_fine = fine.reshape(coarse, block, coarse, block).mean(axis=(1, 3))
    coarse_own = np.asarray(results[str(coarse)]["cell_density"])
    difference = float(np.sqrt(np.mean((coarse_from_fine - coarse_own) ** 2)))
    scale = float(np.sqrt(np.mean(coarse_own ** 2)))
    results["self_convergence"] = {
        "coarse": coarse,
        "fine": ncells,
        "relative_l2_between_meshes": difference / scale,
        "note": "no closed-form solution; this is a mesh-independence indicator only",
    }
    return results


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip-slow", action="store_true",
                        help="skip the vortex convergence and four-quadrant runs")
    args = parser.parse_args(argv)
    torch.set_num_threads(max(1, args.threads))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    report = {}
    started = time.time()
    print("[A] cross-code residual identity (2-D vs independent 1-D DG)")
    report["residual_identity"] = residual_identity()
    for degree, block in report["residual_identity"].items():
        print(
            f"  P{degree}: max |2D-1D| = {max(block['max_abs_difference'].values()):.3e} "
            f"(relative {block['relative_difference']:.3e})"
        )

    print("[B] extruded Sod tube vs exact Riemann solution")
    report["extruded_sod"] = extruded_sod()
    for degree in ("1", "2"):
        block = report["extruded_sod"][degree]
        worst = max(item["relative_l2_percent"] for item in block["errors"].values())
        print(
            f"  P{degree}: worst relative L2 = {worst:.3f}% "
            f"({'PASS' if worst < 3.0 else 'FAIL'} vs 3% gate), {block['steps']} steps"
        )

    if not args.skip_slow:
        print("[C] isentropic vortex convergence")
        report["vortex"] = [
            vortex_convergence(degree=degree, meshes=(16, 24, 32))
            for degree in (1, 2)
        ]
        for block in report["vortex"]:
            last = block["rows"][-1]
            print(
                f"  P{block['degree']}: {last['cells']}^2 rho rel-L2 "
                f"{last['errors']['density']:.3e}, EOC {last.get('eoc', {})}"
            )

        print("[D] four-quadrant Riemann problem (Lax & Liu config 3)")
        report["four_quadrant"] = four_quadrant()
        fine = report["four_quadrant"]["48"]
        print(
            f"  min rho {fine['min_density_quadrature']:.3e}, "
            f"min p {fine['min_pressure_quadrature']:.3e}, finite={fine['finite']}"
        )

    report["environment"] = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "threads": torch.get_num_threads(),
        "python": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
        "wall_seconds": time.time() - started,
    }
    path = output / "verify.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {path} ({path.stat().st_size} bytes, {report['environment']['wall_seconds']:.1f}s)")
    return report


if __name__ == "__main__":
    main()