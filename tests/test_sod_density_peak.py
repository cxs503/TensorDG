"""Pytest tests for Sod shock tube density peak (contact discontinuity) position.

Tests the TensorDG solver's ability to capture the contact discontinuity
position of the Sod (1978) shock tube problem within the <4% tolerance
specified in feature ledger CASE-DG-SOD-SHOCK.

Reference: Sod, G. A. (1978). J. Comput. Phys. 27(1), 1–31.
"""
import numpy as np
import pytest
import torch

from benchmarks.sod_exact import star_state
from benchmarks.euler_sod_1d import solve_sod
from tensordg.euler_dg_1d import solve_euler_dg_1d

# ── Problem constants ──────────────────────────────────────────────────────
GAMMA = 1.4
LEFT = (1.0, 0.0, 1.0)
RIGHT = (0.125, 0.0, 0.1)
FINAL_TIME = 0.2
DIAPHRAGM = 0.5
DOMAIN_LENGTH = 1.0
POSITION_TOLERANCE = 0.04  # 4% of domain length


def _exact_contact():
    _, u_star = star_state()
    return DIAPHRAGM + u_star * FINAL_TIME


def _star_density_avg():
    p_star, _ = star_state()
    rho_star_L = LEFT[0] * (p_star / LEFT[2]) ** (1.0 / GAMMA)
    ratio = p_star / RIGHT[2]
    rho_star_R = RIGHT[0] * (ratio + (GAMMA - 1) / (GAMMA + 1)) / (
        (GAMMA - 1) / (GAMMA + 1) * ratio + 1
    )
    return (rho_star_L + rho_star_R) / 2


def _find_contact_p0(x, density, rho_avg):
    mask = (x > 0.4) & (x < 0.8)
    x_s = x[mask]
    rho_s = density[mask]
    if len(x_s) < 2:
        return 0.5
    idx = np.argmin(np.abs(rho_s - rho_avg))
    return float(x_s[idx])


def _find_contact_ho(x, density, velocity, u_star):
    mask = (x > 0.45) & (x < 0.82)
    x_s = x[mask]
    rho_s = density[mask]
    u_s = velocity[mask]
    if len(x_s) < 3:
        return 0.5
    drho = np.diff(rho_s)
    dx = np.diff(x_s)
    grad = np.abs(drho / dx)
    sorted_idx = np.argsort(grad)[::-1]
    for idx in sorted_idx[:5]:
        if idx < len(u_s) and abs(u_s[idx] - u_star) < 0.2:
            return float(x_s[idx])
    if len(sorted_idx) >= 2:
        return float(x_s[sorted_idx[1]])
    return float(x_s[sorted_idx[0]])


class TestSodDensityPeakPosition:
    """Verify contact discontinuity position for Sod shock tube."""

    def test_p0_contact_position_at_400_cells(self):
        """P0 at N=400 should capture contact within 4%."""
        x_exact = _exact_contact()
        rho_avg = _star_density_avg()

        x, fields, meta = solve_sod(ncells=400, final_time=FINAL_TIME, cfl=0.4)
        x_np = x.detach().cpu().numpy()
        rho_np = fields["density"].detach().cpu().numpy()

        x_contact = _find_contact_p0(x_np, rho_np, rho_avg)
        pos_err = abs(x_contact - x_exact) / DOMAIN_LENGTH

        assert pos_err < POSITION_TOLERANCE, (
            f"P0 N=400 contact position error {pos_err*100:.4f}% >= {POSITION_TOLERANCE*100:.0f}%"
        )

    def test_p1_contact_position_at_50_cells(self):
        """P1 DG at N=50 should capture contact within 4%."""
        x_exact = _exact_contact()
        _, u_star = star_state()

        x, coeff, meta = solve_euler_dg_1d(
            LEFT, RIGHT, ncells=50, degree=1, final_time=FINAL_TIME,
            diaphragm=DIAPHRAGM, cfl=0.12, limiter_stage="final",
        )
        x_np = x.detach().cpu().numpy()
        avg = coeff[:, 0].detach().cpu().numpy()
        rho_np = avg[:, 0]
        u_np = avg[:, 1] / rho_np

        x_contact = _find_contact_ho(x_np, rho_np, u_np, u_star)
        pos_err = abs(x_contact - x_exact) / DOMAIN_LENGTH

        assert pos_err < POSITION_TOLERANCE, (
            f"P1 N=50 contact position error {pos_err*100:.4f}% >= {POSITION_TOLERANCE*100:.0f}%"
        )

    def test_p2_contact_position_at_30_cells(self):
        """P2 DG at N=30 should capture contact within 4%."""
        x_exact = _exact_contact()
        _, u_star = star_state()

        x, coeff, meta = solve_euler_dg_1d(
            LEFT, RIGHT, ncells=30, degree=2, final_time=FINAL_TIME,
            diaphragm=DIAPHRAGM, cfl=0.12, limiter_stage="final",
        )
        x_np = x.detach().cpu().numpy()
        avg = coeff[:, 0].detach().cpu().numpy()
        rho_np = avg[:, 0]
        u_np = avg[:, 1] / rho_np

        x_contact = _find_contact_ho(x_np, rho_np, u_np, u_star)
        pos_err = abs(x_contact - x_exact) / DOMAIN_LENGTH

        assert pos_err < POSITION_TOLERANCE, (
            f"P2 N=30 contact position error {pos_err*100:.4f}% >= {POSITION_TOLERANCE*100:.0f}%"
        )

    def test_p3_contact_position_at_20_cells(self):
        """P3 DG at N=20 should capture contact within 4%."""
        x_exact = _exact_contact()
        _, u_star = star_state()

        x, coeff, meta = solve_euler_dg_1d(
            LEFT, RIGHT, ncells=20, degree=3, final_time=FINAL_TIME,
            diaphragm=DIAPHRAGM, cfl=0.12, limiter_stage="final",
        )
        x_np = x.detach().cpu().numpy()
        avg = coeff[:, 0].detach().cpu().numpy()
        rho_np = avg[:, 0]
        u_np = avg[:, 1] / rho_np

        x_contact = _find_contact_ho(x_np, rho_np, u_np, u_star)
        pos_err = abs(x_contact - x_exact) / DOMAIN_LENGTH

        assert pos_err < POSITION_TOLERANCE, (
            f"P3 N=20 contact position error {pos_err*100:.4f}% >= {POSITION_TOLERANCE*100:.0f}%"
        )

    def test_p2_contact_position_at_multiple_resolutions(self):
        """P2 DG contact position should be within tolerance at multiple resolutions."""
        x_exact = _exact_contact()
        _, u_star = star_state()

        for ncells in [30, 60, 120]:
            x, coeff, _ = solve_euler_dg_1d(
                LEFT, RIGHT, ncells=ncells, degree=2, final_time=FINAL_TIME,
                diaphragm=DIAPHRAGM, cfl=0.12, limiter_stage="final",
            )
            x_np = x.detach().cpu().numpy()
            avg = coeff[:, 0].detach().cpu().numpy()
            rho_np = avg[:, 0]
            u_np = avg[:, 1] / rho_np

            x_contact = _find_contact_ho(x_np, rho_np, u_np, u_star)
            pos_err = abs(x_contact - x_exact) / DOMAIN_LENGTH

            assert pos_err < POSITION_TOLERANCE, (
                f"P2 N={ncells} contact position error {pos_err*100:.4f}% >= {POSITION_TOLERANCE*100:.0f}%"
            )

    def test_density_peak_value_near_exact(self):
        """Maximum density in numerical solution should be close to 1.0."""
        x, fields, meta = solve_sod(ncells=400, final_time=FINAL_TIME, cfl=0.4)
        rho_max = fields["density"].max().item()
        # The exact max density is 1.0 (left state)
        assert abs(rho_max - 1.0) < 0.05, f"P0 density max {rho_max:.6f} too far from 1.0"

    def test_star_state_reference(self):
        """Verify the star state matches Sod 1978 reference values."""
        p_star, u_star = star_state()
        # Reference values from Sod (1978) and verified by exact Riemann solver
        assert abs(p_star - 0.30313017805064685) < 1e-10
        assert abs(u_star - 0.9274526200489499) < 1e-10

    def test_exact_contact_position(self):
        """Verify the exact contact discontinuity position."""
        x_contact = _exact_contact()
        # x_contact = 0.5 + u* * 0.2
        expected = 0.5 + 0.9274526200489499 * 0.2
        assert abs(x_contact - expected) < 1e-10
