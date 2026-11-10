"""Measure the Audusse reconstruction numbers quoted in the documentation.

Writes a JSON artifact to ``validation/swe_thacker/hydrostatic_recon_results.json``
so that every number in ``docs/tensordg-swe-hydrostatic-wellbalanced.md`` is
programmatically traceable (no hand-copied values).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Allow a direct ``python validation/swe_thacker/run_hydrostatic_recon.py``
# from the repository checkout (the package is not always installed).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402  (import after the sys.path bootstrap)

from tensordg.swe_1d import (  # noqa: E402
    _lgl_transform, reconstruct_fields, solve_swe_1d)

ETA0 = 2.0
STEP_BED = 0.5
G = 9.81
DOMAIN = (-2.0, 2.0)


def cell_step_bed(ncells: int, degree: int):
    """Cell-local nodal bed with a jump at x = 0 (left 0.5 m, right 0 m)."""
    lgl = _lgl_transform(degree, torch.device("cpu"), torch.float64)[1]
    dx = (DOMAIN[1] - DOMAIN[0]) / ncells
    xc = DOMAIN[0] + (torch.arange(ncells, dtype=torch.float64) + 0.5) * dx
    node_x = xc[:, None] + 0.5 * dx * lgl[None, :]
    left = (torch.arange(ncells) < ncells // 2)[:, None]
    zb = torch.where(left, torch.full_like(node_x, STEP_BED),
                     torch.zeros_like(node_x))
    return node_x, zb


def rest_state(ncells: int, degree: int, final_time: float = 1.0,
               hydrostatic: bool = True) -> dict:
    node_x, zb_nodes = cell_step_bed(ncells, degree)
    h_nodes = (ETA0 - zb_nodes).clamp_min(0.0)
    t0 = time.perf_counter()
    x, coeff, zb_faces, meta = solve_swe_1d(
        h_nodes, torch.zeros_like(zb_nodes), zb_nodes, ncells=ncells,
        degree=degree, final_time=final_time, domain=DOMAIN, cfl=0.3,
        rk_order=3, well_balanced=True, hydrostatic=hydrostatic)
    wall = time.perf_counter() - t0
    h, u, eta = reconstruct_fields(coeff, zb_faces,
                                   zb_center=meta["zb_modes"][:, 0])
    wet = h > 1e-6
    return {
        "ncells": ncells, "degree": degree, "final_time": final_time,
        "hydrostatic_reconstruction": bool(meta["hydrostatic_reconstruction"]),
        "eta_error_max_wet": float((eta[wet] - ETA0).abs().max()),
        "u_max": float(u.abs().max()),
        "hu_modal_max": float(coeff[:, :, 1].abs().max()),
        "wall_time_s": round(wall, 3), "steps": meta["steps"],
        "reached_final_time": bool(meta["reached_final_time"]),
    }


def dry_wall_case(ncells: int, degree: int, final_time: float = 1.0) -> dict:
    """Water (eta = 2) impounded against a bed that rises to 2.5 m."""
    lgl = _lgl_transform(degree, torch.device("cpu"), torch.float64)[1]
    dx = (DOMAIN[1] - DOMAIN[0]) / ncells
    xc = DOMAIN[0] + (torch.arange(ncells, dtype=torch.float64) + 0.5) * dx
    node_x = xc[:, None] + 0.5 * dx * lgl[None, :]
    left = (torch.arange(ncells) < ncells // 2)[:, None]
    zb = torch.where(left, torch.full_like(node_x, 0.5),
                     torch.full_like(node_x, 2.5))
    h = torch.where(left, torch.full_like(node_x, 1.5),
                    torch.zeros_like(node_x))
    x, coeff, zb_faces, meta = solve_swe_1d(
        h, torch.zeros_like(h), zb, ncells=ncells, degree=degree,
        final_time=final_time, domain=DOMAIN, cfl=0.3, rk_order=3,
        well_balanced=True)
    hh, u, eta = reconstruct_fields(coeff, zb_faces,
                                    zb_center=meta["zb_modes"][:, 0])
    dry = hh < 1e-6
    return {
        "ncells": ncells, "degree": degree, "final_time": final_time,
        "n_dry_cells": int(dry.sum()), "n_expected_dry": ncells // 2,
        "max_depth_on_dry_side": float(hh[dry].max()),
        "min_depth_on_wet_side": float(hh[~dry].min()),
        "eta_error_max_wet": float((eta[~dry] - ETA0).abs().max()),
        "u_max": float(u.abs().max()),
        "mass_initial": 1.5 * (DOMAIN[1] - DOMAIN[0]) / 2,
        "reached_final_time": bool(meta["reached_final_time"]),
    }


def dam_break_over_step(ncells: int = 100, degree: int = 1,
                        final_time: float = 0.3) -> dict:
    node_x, zb_nodes = cell_step_bed(ncells, degree)
    left = (torch.arange(ncells) < ncells // 2)[:, None]
    h_nodes = torch.where(left, torch.full_like(node_x, 2.0),
                          torch.full_like(node_x, 1.0))
    x, coeff, zb_faces, meta = solve_swe_1d(
        h_nodes, torch.zeros_like(node_x), zb_nodes, ncells=ncells,
        degree=degree, final_time=final_time, domain=DOMAIN, cfl=0.3,
        rk_order=3, well_balanced=True)
    h, u, _ = reconstruct_fields(coeff, zb_faces,
                                 zb_center=meta["zb_modes"][:, 0])
    dx = (DOMAIN[1] - DOMAIN[0]) / ncells
    return {
        "ncells": ncells, "degree": degree, "final_time": final_time,
        "min_depth": float(h.min()), "max_speed": float(u.abs().max()),
        "mass_final": float(h.sum()) * dx, "mass_initial_exact": 6.0,
        "mass_rel_error": abs(float(h.sum()) * dx - 6.0) / 6.0,
        "reached_final_time": bool(meta["reached_final_time"]),
    }


def thacker_moving_solution(ncells: int = 20, final_time: float = 0.5) -> list:
    """Accuracy against the Thacker (1981) analytic solution, moving free surface.

    Same closure, but the solution is *not* at rest: this is the guard that the
    reconstruction does not degrade the smooth-bed accuracy (it is the identity
    there) and that the analytic boundary data is honoured.
    """
    from tensordg import swe_exact

    h0, amp, half = 1.0, 0.5, 1.0

    def bed(xx):
        return torch.as_tensor(swe_exact.thacker_parabolic_bed(xx.numpy(), h0, half),
                               dtype=xx.dtype)

    def init_h(xx):
        h, _ = swe_exact.thacker_planar_basin_initial(xx.numpy(), h0, amp, half)
        return torch.as_tensor(h, dtype=xx.dtype)

    def left(t):
        return torch.as_tensor(swe_exact.thacker_planar_basin_boundary_flux(
            t, h0, amp, half, G)[0])

    def right(t):
        return torch.as_tensor(swe_exact.thacker_planar_basin_boundary_flux(
            t, h0, amp, half, G)[1])

    out = []
    for degree in (1, 2):
        x, coeff, zb_faces, meta = solve_swe_1d(
            init_h, lambda xx: torch.zeros_like(xx), bed, ncells=ncells,
            degree=degree, final_time=final_time, domain=(-half, half), cfl=0.3,
            limiter=None, rk_order=3, well_balanced=True,
            bc_left_flux=left, bc_right_flux=right)
        h, u, eta = reconstruct_fields(coeff, zb_faces,
                                       zb_center=meta["zb_modes"][:, 0])
        h_ex, u_ex, eta_ex = swe_exact.thacker_planar_basin(
            x.numpy(), final_time, h0, amp, half, G)
        h_ex = torch.as_tensor(h_ex)
        u_ex = torch.as_tensor(u_ex)
        eta_ex = torch.as_tensor(eta_ex)

        def rel(a, b):
            return float(torch.sqrt(torch.mean((a - b) ** 2)) / b.abs().max()) * 100.0

        out.append({
            "degree": degree, "ncells": ncells, "final_time": final_time,
            "eta_rel_L2_pct": rel(eta, eta_ex), "h_rel_L2_pct": rel(h, h_ex),
            "u_rel_L2_pct": rel(u, u_ex), "steps": meta["steps"],
        })
    return out


def main() -> None:
    out = {
        "description": (
            "Audusse hydrostatic reconstruction: lake at rest across a "
            "submerged bed step (z_b = 0.5 m left of x=0, 0 m right, eta = 2 m), "
            "a wet/dry bed wall (z_b = 2.5 m right of x=0), and a dam break "
            "across the step.  Domain (-2, 2) m, cfl = 0.3, SSP-RK3."),
        "gravity": G,
        "rest_state_across_step": [],
        "rest_state_without_reconstruction": [],
        "wet_dry_wall": [],
        "dam_break_over_step": [],
        "thacker_moving_solution_accuracy": [],
    }
    for degree in (1, 2):
        for ncells in (32, 64):
            out["rest_state_across_step"].append(rest_state(ncells, degree))
            out["rest_state_without_reconstruction"].append(
                rest_state(ncells, degree, hydrostatic=False))
            out["wet_dry_wall"].append(dry_wall_case(ncells, degree))
    out["dam_break_over_step"].append(dam_break_over_step())
    out["thacker_moving_solution_accuracy"] = thacker_moving_solution()
    Path("validation/swe_thacker/hydrostatic_recon_results.json").write_text(
        json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
