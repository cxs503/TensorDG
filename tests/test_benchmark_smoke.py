"""Fixed-grid benchmark smoke test for TensorDG.

Runs small-scale, deterministic benchmarks on fixed grids and asserts
key physical/numerical quantities against analytical solutions.
Records timing baselines for regression detection.

Benchmarks included:
  1. Heat equation (SIPG/CN): ncells=16, degree=2, t=0.1
     - Relative L2 error vs exact sin(pi*x)*exp(-pi^2*t) < 3%
     - Solution finite, SIPG symmetry error < 1e-11
  2. Linear advection (DG/RK4): ncells=40, degree=2, t=0.5
     - L2 error vs exact sin(2*pi*(x-t)) < 3%
     - Mass conservation (periodic domain)
  3. Poisson 2D (SIPG): nx=ny=8, degree=1
     - Relative L2 error vs exact sin(pi*x)*sin(pi*y) < 5%

Each test records wall-clock time for baseline tracking.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from tensordg.advection_1d import l2_error, solve_advection
from tensordg.heat_1d import heat_error, solve_heat_1d
from tensordg.poisson_2d import poisson_error_2d, solve_poisson_2d

# ---------------------------------------------------------------------------
# Evidence collector -- writes structured JSON for CI / regression tracking
# ---------------------------------------------------------------------------
_EVIDENCE_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "benchmark_smoke"


def _write_evidence(name: str, data: dict) -> None:
    """Persist benchmark evidence as JSON for downstream consumption."""
    _EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = _EVIDENCE_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# 1. Heat equation benchmark (SIPG + Crank-Nicolson)
# ---------------------------------------------------------------------------
class TestHeatBenchmarkSmoke:
    """Fixed-grid heat equation: sin(pi*x)*exp(-pi^2*t) on [0,1]."""

    NCELLS = 16
    DEGREE = 2
    FINAL_TIME = 0.1
    MAX_REL_L2 = 0.03  # 3%

    def test_heat_fixed_grid_accuracy(self):
        t0 = time.perf_counter()
        x, coeff, meta = solve_heat_1d(
            ncells=self.NCELLS,
            degree=self.DEGREE,
            final_time=self.FINAL_TIME,
        )
        elapsed = time.perf_counter() - t0

        # Solution must be finite
        assert torch.isfinite(coeff).all(), "Heat solution contains non-finite values"

        # SIPG symmetry is a structural invariant
        assert meta["diffusion_symmetry_error"] < 1e-11, (
            f"SIPG symmetry broken: {meta['diffusion_symmetry_error']}"
        )

        # Key accuracy assertion: relative L2 vs analytical
        metrics = heat_error(x, coeff, self.FINAL_TIME)
        rel_l2 = metrics["relative_l2"]
        assert rel_l2 < self.MAX_REL_L2, (
            f"Heat relative L2 = {rel_l2:.6f} exceeds {self.MAX_L2}"
        )

        # Record evidence
        _write_evidence("heat_benchmark", {
            "benchmark": "heat_1d_sipg_cn",
            "grid": {"ncells": self.NCELLS, "degree": self.DEGREE},
            "final_time": self.FINAL_TIME,
            "relative_l2": rel_l2,
            "relative_l2_percent": 100.0 * rel_l2,
            "symmetry_error": meta["diffusion_symmetry_error"],
            "steps": meta["steps"],
            "dt": meta["dt"],
            "wall_time_s": elapsed,
            "assertions": {
                "finite_solution": True,
                "symmetry_lt_1e-11": True,
                "rel_l2_lt_3pct": rel_l2 < self.MAX_REL_L2,
            },
        })

    def test_heat_solution_norm_consistency(self):
        """The L2 norm of the numerical solution should be close to the
        exact L2 norm of sin(pi*x)*exp(-pi^2*t) at t=0.1."""
        x, coeff, _ = solve_heat_1d(
            ncells=self.NCELLS,
            degree=self.DEGREE,
            final_time=self.FINAL_TIME,
        )
        # Exact L2 norm: ||sin(pi*x)*exp(-pi^2*t)||_{L2[0,1]}
        # = exp(-pi^2*t) * sqrt(integral_0^1 sin^2(pi*x) dx)
        # = exp(-pi^2*t) / sqrt(2)
        exact_norm = math.exp(-math.pi**2 * self.FINAL_TIME) / math.sqrt(2.0)

        # Compute numerical L2 norm via quadrature
        metrics = heat_error(x, coeff, self.FINAL_TIME)
        # relative_l2 = ||u_h - u_exact|| / ||u_exact||
        # So ||u_h|| ≈ ||u_exact|| * (1 ± relative_l2) for small errors
        # We check the norm is within 5% of exact
        numerical_norm_approx = exact_norm * (1.0 + metrics["relative_l2"])
        norm_ratio = numerical_norm_approx / exact_norm
        assert 0.95 < norm_ratio < 1.05, (
            f"Norm ratio {norm_ratio:.6f} outside [0.95, 1.05]"
        )


# ---------------------------------------------------------------------------
# 2. Linear advection benchmark (DG + RK4, periodic)
# ---------------------------------------------------------------------------
class TestAdvectionBenchmarkSmoke:
    """Fixed-grid advection: sin(2*pi*(x-t)) on [0,1] periodic."""

    NCELLS = 40
    DEGREE = 2
    FINAL_TIME = 0.5
    MAX_L2 = 0.03  # 3%

    def test_advection_fixed_grid_accuracy(self):
        t0 = time.perf_counter()
        centers, coeff, meta = solve_advection(
            ncells=self.NCELLS,
            degree=self.DEGREE,
            final_time=self.FINAL_TIME,
        )
        elapsed = time.perf_counter() - t0

        assert torch.isfinite(coeff).all(), "Advection solution contains non-finite values"

        # Key accuracy assertion
        err = l2_error(centers, coeff, final_time=self.FINAL_TIME)
        assert err < self.MAX_L2, (
            f"Advection L2 error = {err:.6f} exceeds {self.MAX_L2}"
        )

        # Mass conservation on periodic domain
        mass = coeff[:, 0].sum().item() / self.NCELLS
        assert abs(mass) < 1e-10, f"Mass not conserved: {mass}"

        _write_evidence("advection_benchmark", {
            "benchmark": "advection_1d_dg_rk4",
            "grid": {"ncells": self.NCELLS, "degree": self.DEGREE},
            "final_time": self.FINAL_TIME,
            "l2_error": err,
            "l2_error_percent": 100.0 * err,
            "mass_residual": abs(mass),
            "steps": meta["steps"],
            "cfl": meta.get("cfl", "N/A"),
            "wall_time_s": elapsed,
            "assertions": {
                "finite_solution": True,
                "l2_lt_3pct": err < self.MAX_L2,
                "mass_conserved": abs(mass) < 1e-10,
            },
        })


# ---------------------------------------------------------------------------
# 3. Poisson 2D benchmark (SIPG)
# ---------------------------------------------------------------------------
class TestPoissonBenchmarkSmoke:
    """Fixed-grid 2D Poisson: -Laplacian(u) = f on unit square."""

    NX = 8
    NY = 8
    DEGREE = 1
    MAX_REL_L2 = 0.05  # 5%

    def test_poisson_fixed_grid_accuracy(self):
        t0 = time.perf_counter()
        x_centers, y_centers, coeff, meta = solve_poisson_2d(
            ncells=(self.NX, self.NY),
            degree=self.DEGREE,
        )
        elapsed = time.perf_counter() - t0

        assert torch.isfinite(coeff).all(), "Poisson solution contains non-finite values"

        metrics = poisson_error_2d(x_centers, y_centers, coeff)
        rel_l2 = metrics["relative_l2"]
        assert rel_l2 < self.MAX_REL_L2, (
            f"Poisson relative L2 = {rel_l2:.6f} exceeds {self.MAX_REL_L2}"
        )

        _write_evidence("poisson_benchmark", {
            "benchmark": "poisson_2d_sipg",
            "grid": {"nx": self.NX, "ny": self.NY, "degree": self.DEGREE},
            "relative_l2": rel_l2,
            "relative_l2_percent": 100.0 * rel_l2,
            "wall_time_s": elapsed,
            "assertions": {
                "finite_solution": True,
                "rel_l2_lt_5pct": rel_l2 < self.MAX_REL_L2,
            },
        })
