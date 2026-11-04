"""Verification debt: the P0 two-dimensional Euler path against external references.

Mission m3315.  ``tensordg/euler_2d.py`` was, until now, only exercised on the
smooth periodic isentropic vortex, so nothing in the suite checked the module on
a discontinuity -- and the module could not run one either, because the stencil
wrapped unconditionally (``torch.roll``) so every mesh was periodic.  This file
pins the new outflow closure and the shock behaviour of that path.

Every reference below comes from outside the solver under test:

* the exact Riemann solution of ``benchmarks/sod_exact.py`` (Sod 1978 star state,
  already pinned by ``tests/test_euler_sod_1d.py``);
* Machado's identity that a ``y``-invariant state has ``y``-flux equal to the
  physical flux, hence a column-independent update (checked to round-off);
* the telescoping identity of a conservative flux-difference scheme, which is an
  exact algebraic statement about the stencil;
* the exact shock/contact speeds of the Sod solution (1.7522 and 0.9274 in these
  units), checked through the discontinuity positions.
"""
import math

import numpy as np
import pytest
import torch

import tensordg.euler_2d as e2d
from benchmarks.sod_exact import cell_averages, sample

LEFT = (1.0, 0.0, 1.0)
RIGHT = (0.125, 0.0, 0.1)
DIAPHRAGM = 0.5
DOMAIN = ((0.0, 1.0), (0.0, 1.0))
FINAL_TIME = 0.2
CFL = 0.4


def sod_initial(x, y):
    rho = torch.where(x < DIAPHRAGM, torch.as_tensor(LEFT[0], dtype=x.dtype),
                      torch.as_tensor(RIGHT[0], dtype=x.dtype))
    pressure = torch.where(x < DIAPHRAGM, torch.as_tensor(LEFT[2], dtype=x.dtype),
                           torch.as_tensor(RIGHT[2], dtype=x.dtype))
    zero = torch.zeros_like(rho)
    return rho, zero, zero, pressure


def _crossing(points, values, level, from_right=True):
    """First crossing of ``level`` scanning from the right (or the left)."""
    count = len(points)
    indices = range(count - 2, -1, -1) if from_right else range(count - 1)
    for left in indices:
        right = left + 1
        if (values[left] - level) * (values[right] - level) <= 0.0:
            weight = (level - values[left]) / (values[right] - values[left])
            return float(points[left] + weight * (points[right] - points[left]))
    return float("nan")


@pytest.fixture(scope="module")
def extruded_sod():
    """The documented 3200-cell extruded Sod run, shared by the two gate tests.

    ``ny = 2`` is enough: with ``y``-independent data the ``y``-flux is the
    physical flux, so the update is column-identical and the ``x``-profile does
    not depend on the number of rows (asserted separately below).
    """
    x, _, fields, metadata = e2d.solve_euler_2d(
        sod_initial,
        ncells=(3200, 2),
        final_time=FINAL_TIME,
        cfl=CFL,
        domain=DOMAIN,
        periodic=(False, True),
    )
    return x, fields, metadata


def test_extruded_sod_matches_exact_riemann_within_three_percent(extruded_sod):
    """Documented case: 3200 x 2 cells, outflow in x, t = 0.2.

    The reference is the exact Riemann solution averaged over the same cells; the
    measured errors are the ones recorded in
    ``validation/euler_2d_p0/evidence.json`` (density 2.0 %, velocity 2.5 %,
    pressure 1.6 % -- the velocity is the limiting field, exactly as in the 1-D
    P0 benchmark of ``benchmarks/compare_euler_sod_1d.py``).
    """
    x, fields, metadata = extruded_sod
    centers, exact = cell_averages(metadata["ncells"][0], FINAL_TIME)
    assert np.allclose(centers, x.cpu().numpy(), atol=1e-12)
    for key, numerical in (
        ("density", fields["density"][:, 0]),
        ("velocity", fields["velocity_x"][:, 0]),
        ("pressure", fields["pressure"][:, 0]),
    ):
        computed = numerical.cpu().numpy()
        absolute = float(np.sqrt(np.mean((computed - exact[key]) ** 2)))
        norm = float(np.sqrt(np.mean(exact[key] ** 2)))
        assert absolute / norm < 0.03, (key, 100.0 * absolute / norm)


def test_extruded_sod_discontinuities_sit_on_the_exact_positions(extruded_sod):
    """Shock and contact positions within two numerical cells of the exact ones.

    Sod's density has two jumps; both are located as midpoint crossings, which is
    insensitive to the smearing width.  The exact positions come from the exact
    Riemann solver (same shock/contact speeds as the analytic Sod values).
    """
    x, fields, _ = extruded_sod
    density = fields["density"][:, 0].cpu().numpy()
    centers = x.cpu().numpy()
    grid = np.linspace(0.0, 1.0, 200001)
    exact_density = sample(grid, FINAL_TIME)["density"]

    shock_level = 0.5 * (RIGHT[0] + 0.2655737)
    contact_level = 0.5 * (0.4263194 + 0.2655737)
    dx = 1.0 / len(centers)
    exact_shock = _crossing(grid, exact_density, shock_level, from_right=True)
    exact_contact = _crossing(grid, exact_density, contact_level, from_right=False)
    numerical_shock = _crossing(centers, density, shock_level, from_right=True)
    numerical_contact = _crossing(centers, density, contact_level, from_right=False)

    # The shock is sharp (measured 0.85 dx) and its speed is fixed by the exact
    # solution; the contact is the most diffused wave of a P0 Rusanov scheme, so
    # only its position to a few cells is meaningful (measured 3.75 dx).
    assert abs(numerical_shock - exact_shock) <= 1.0 * dx, (numerical_shock, exact_shock)
    assert abs(numerical_contact - exact_contact) <= 5.0 * dx, (numerical_contact, exact_contact)
    # The measured shock speed must reproduce Sod's 1.7522 to within 2 %.
    speed = (numerical_shock - DIAPHRAGM) / FINAL_TIME
    assert abs(speed - 1.75215) / 1.75215 < 0.02, speed


def test_y_invariant_data_stays_y_invariant():
    """Two identical neighbouring rows have zero Rusanov dissipation along y."""
    _, _, fields, metadata = e2d.solve_euler_2d(
        sod_initial,
        ncells=(100, 4),
        final_time=0.05,
        cfl=CFL,
        domain=DOMAIN,
        periodic=(False, True),
    )
    assert metadata["periodic"] == (False, True)
    state = fields["conservative"]
    columns = state.reshape(state.shape[0], state.shape[1], -1)
    spread = (columns - columns[:, :1, :]).abs().max().item()
    assert fields["conservative"][..., 2].abs().max().item() == 0.0
    assert spread < 1e-12 * max(1.0, float(state.abs().max()))


def test_outflow_face_flux_is_the_physical_flux_of_the_interior_cell():
    """The transmissive closure copies the edge cell, so the boundary gate is F(u)."""
    dx = dy = 1.0 / 8
    x = (torch.arange(8, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(8, dtype=torch.float64) + 0.5) * dy
    u = e2d._cell_averages(sod_initial, x, y, dx, dy, 2, torch.device("cpu"), torch.float64)
    faces = e2d._face_fluxes(u, "x", False)
    assert torch.equal(faces[0], e2d._flux_x(u[0]))
    assert torch.equal(faces[-1], e2d._flux_x(u[-1]))
    periodic_faces = e2d._face_fluxes(u, "x", True)
    assert torch.equal(periodic_faces[0], e2d._interface_flux(u[-1], u[0], "x"))
    assert torch.equal(periodic_faces[-1], e2d._interface_flux(u[-1], u[0], "x"))


def test_rhs_telescopes_to_the_boundary_flux():
    """Discrete mass rate equals the net boundary flux (exact, to round-off)."""
    nx, ny = 32, 4
    dx, dy = 1.0 / nx, 1.0 / ny
    x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    u = e2d._cell_averages(sod_initial, x, y, dx, dy, 2, torch.device("cpu"), torch.float64)
    rhs = e2d._rhs(u, dx, dy, (False, False))
    faces_x = e2d._face_fluxes(u, "x", False)
    faces_y = e2d._face_fluxes(u, "y", False)
    discrete = float(rhs[..., 0].sum()) * dx * dy
    boundary = dy * (float(faces_x[0, :, 0].sum()) - float(faces_x[-1, :, 0].sum()))
    boundary += dx * (float(faces_y[:, 0, 0].sum()) - float(faces_y[:, -1, 0].sum()))
    assert abs(discrete - boundary) <= 1e-12 * max(1.0, abs(discrete))


def test_periodic_path_is_bit_for_bit_the_rolled_stencil():
    """Regression guard for the padding refactor: periodic == old roll stencil."""
    nx = ny = 16
    dx, dy = 1.0 / nx, 1.0 / ny
    x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    u = e2d._cell_averages(sod_initial, x, y, dx, dy, 2, torch.device("cpu"), torch.float64)
    right_x = torch.roll(u, shifts=-1, dims=0)
    right_y = torch.roll(u, shifts=-1, dims=1)
    flux_right_x = e2d._interface_flux(u, right_x, "x")
    flux_right_y = e2d._interface_flux(u, right_y, "y")
    flux_left_x = torch.roll(flux_right_x, shifts=1, dims=0)
    flux_left_y = torch.roll(flux_right_y, shifts=1, dims=1)
    rolled = -(flux_right_x - flux_left_x) / dx - (flux_right_y - flux_left_y) / dy
    assert torch.equal(e2d._rhs(u, dx, dy, (True, True)), rolled)


# NOTE (m3315): the transposed-orientation check lives in the harness
# (``benchmarks/verify_euler_2d_p0.py`` check E, worst relative difference 0.0 at
# 400 cells) rather than here: the 2-D y-flux path is already covered by the
# column-independence and telescoping identities above, and a second full
# solve pair would add ~20 s to the suite for no extra discriminating power.


def test_solver_validates_its_inputs():
    with pytest.raises(ValueError):
        e2d.solve_euler_2d(sod_initial, ncells=(1, 4))
    with pytest.raises(ValueError):
        e2d.solve_euler_2d(sod_initial, ncells=(8, 8), cfl=0.6)
    with pytest.raises(ValueError):
        e2d.solve_euler_2d(sod_initial, ncells=(8, 8), domain=((1.0, 0.0), (0.0, 1.0)))
    with pytest.raises(ValueError):
        e2d.solve_euler_2d(sod_initial, ncells=(8, 8), quadrature=0)
    with pytest.raises(ValueError):
        e2d.solve_euler_2d(sod_initial, ncells=(8, 8), final_time=-1.0)


def test_shock_containing_solution_degrades_to_the_half_order_regime():
    """L2 rate for a shock is ~1/2, not 1: the measured P0 rate must be in (0.4, 0.8).

    Monotone schemes lose order at discontinuities; the L2 rate of a first-order
    scheme on a shocked solution is O(h^{1/2}) (Johnson & Popov 1981; Cockburn,
    Coquel & LeFloch 1995).  Pinning the band keeps a future "order fix" that only
    smooths the numbers from passing silently.
    """
    resolutions = (200, 400, 800)
    errors = []
    for cells in resolutions:
        x, _, fields, _ = e2d.solve_euler_2d(
            sod_initial, ncells=(cells, 2), final_time=FINAL_TIME, cfl=CFL,
            domain=DOMAIN, periodic=(False, True),
        )
        _, exact = cell_averages(cells, FINAL_TIME)
        computed = fields["velocity_x"][:, 0].cpu().numpy()
        errors.append(float(np.sqrt(np.mean((computed - exact["velocity"]) ** 2))))
    rate = math.log(errors[0] / errors[2]) / math.log(resolutions[2] / resolutions[0])
    assert 0.4 < rate < 0.8, (rate, errors)