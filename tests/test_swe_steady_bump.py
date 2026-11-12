"""Verification-debt closure: *moving* steady state over a curved bed.

Before this file, the well-balanced claims in :mod:`tensordg.swe_1d` were only
tested on *rest* states (lake at rest, wet/dry shoreline, Thacker sloshing).  The
classical Goutal & Maurel (1997) / Garcia-Navarro & Vazquez-Cendon (2000)
subcritical bump case supplies the missing anchor: a smooth *moving* steady
solution with ``q = hu != 0`` over a parabolic bed.

Three things are asserted:

1. the exact solution really solves the steady shallow-water equations
   (Bernoulli identity, discharge constancy, critical margin, and a
   second-order finite-difference residual of the steady momentum balance);
2. the semi-discrete defect ``||dU/dt||/||U||`` of the production operator on
   that exact state is *machine zero in mass* and decays as ``O(dx^p)`` in
   momentum -- with the well-balanced nodal path at least 2x smaller than the
   legacy L2-projection path.  A lake-at-rest control shows the metric is not
   vacuous (there the defect is round-off);
3. time marching from the exact steady state settles on a discrete steady state
   whose relative depth error is < 3% (measured 0.022%/0.090% at N=100/N=50) and
   converges at second order, so the non-zero defect does not accumulate into a
   steady bias.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordg import swe_1d
from tensordg.swe_steady_bump import (
    DEPTH_OUT,
    DISCHARGE,
    DOMAIN,
    bump_bed,
    bump_bed_slope,
    bump_boundary_flux,
    bump_steady_state,
    critical_depth,
    steady_state_defect,
)

G = 9.81
REL = 1e-3


def _h(x):
    return bump_steady_state(x)["h"]


def _u(x):
    return bump_steady_state(x)["u"]


def test_exact_solution_identities():
    """The closed form satisfies every steady-state identity used downstream."""
    x = torch.linspace(DOMAIN[0], DOMAIN[1], 2001, dtype=torch.float64)
    st = bump_steady_state(x)
    # total head = free surface + velocity head (the bed is inside eta)
    head = st["eta"] + DISCHARGE ** 2 / (2 * G * st["h"] ** 2)
    assert float((head - st["head"]).abs().max()) < 1e-10 * st["head"]
    q = st["h"] * st["u"]
    assert float((q - DISCHARGE).abs().max()) < 1e-10 * DISCHARGE
    assert float((st["u"] / torch.sqrt(G * st["h"])).max()) < 1.0
    # documented values of the classical case
    assert abs(st["head"] - 2.248935) < 1e-6
    assert abs(float(bump_steady_state(torch.tensor([10.0]))["h"][0])
               - 1.707347) < 1e-5      # depth at the crest
    assert abs(st["critical_margin"] - 0.161741) < 1e-5
    assert abs(critical_depth() - 1.258129) < 1e-5


def test_exact_solution_satisfies_steady_momentum_balance():
    """Central differences of the exact solution balance flux vs bed source."""
    def residual(ncells):
        x = torch.linspace(DOMAIN[0], DOMAIN[1], ncells, dtype=torch.float64)
        st = bump_steady_state(x)
        # restrict to the smooth part of the bed: dz_b/dx has kinks at 8 and 12
        keep = (x > 8.1) & (x < 11.9)
        mom = DISCHARGE ** 2 / st["h"] + 0.5 * G * st["h"] ** 2
        dmom = (mom[2:] - mom[:-2]) / (x[2:] - x[:-2])
        src = G * st["h"][1:-1] * bump_bed_slope(x[1:-1])
        res = (dmom + src).abs()[keep[1:-1]]
        scale = (G * st["h"] * bump_bed_slope(x)).abs().max()
        return float(res.max() / scale)

    coarse, fine = residual(501), residual(2001)
    assert fine < 1e-4
    assert math.log(coarse / fine) / math.log(4.0) > 1.8       # O(dx^2) FD


def test_choked_parameters_are_rejected():
    """A flow that cannot pass the bump subcritically must raise, not silently drift."""
    with pytest.raises(ValueError):
        bump_steady_state(torch.tensor([10.0]), discharge=1.53, depth_out=0.66)


def _h_flat(x):
    return torch.ones_like(torch.as_tensor(x, dtype=torch.float64)) * 2.0


def _zero(x):
    return torch.zeros_like(torch.as_tensor(x, dtype=torch.float64))


def test_lake_at_rest_is_an_exact_discrete_fixed_point():
    """Negative control: at rest the same defect metric returns round-off."""
    flat = lambda x: torch.zeros_like(torch.as_tensor(x, dtype=torch.float64))
    d = steady_state_defect(_h_flat, _zero, flat, ncells=50, degree=1,
                            well_balanced=True)
    assert d["defect_h"] < 1e-13
    assert d["defect_hu_linf"] < 1e-13     # absolute: momentum flux is exactly balanced


@pytest.mark.parametrize("degree,lo,hi", [(1, 0.85, 1.15), (2, 1.8, 2.2)])
def test_momentum_defect_order_is_p(degree, lo, hi):
    """``||dU/dt||/||U||`` decays as ``O(dx^p)`` for the well-balanced path."""
    runs = [steady_state_defect(_h, _u, bump_bed, ncells=n, degree=degree,
                                well_balanced=True) for n in (50, 100, 200)]
    for run in runs:
        assert run["defect_h"] < 1e-12                 # mass: exact balance
    orders = [math.log(runs[i]["defect_hu"] / runs[i + 1]["defect_hu"]) / math.log(2.0)
              for i in range(2)]
    for order in orders:
        assert lo <= order <= hi, f"orders={orders}"


def test_well_balanced_beats_legacy_projection():
    """The nodal well-balanced path must not be worse than the L2-projection path."""
    wb = steady_state_defect(_h, _u, bump_bed, ncells=100, degree=1,
                             well_balanced=True)
    legacy = steady_state_defect(_h, _u, bump_bed, ncells=100, degree=1,
                                 well_balanced=False, zb_slope=bump_bed_slope)
    assert wb["defect_hu"] < 0.5 * legacy["defect_hu"]
    # ... and it is the smallest of the three paths measured in the driver
    face = steady_state_defect(_h, _u, bump_bed, ncells=100, degree=1,
                               well_balanced=False)
    assert wb["defect_hu"] < face["defect_hu"]


def _steady_run(ncells, degree, final_time):
    x, coeff, zb_faces, meta = swe_1d.solve_swe_1d(
        _h, _u, bump_bed, ncells=ncells, degree=degree, final_time=final_time,
        domain=DOMAIN, cfl=0.4, rk_order=3, well_balanced=True,
        bc_left_flux=bump_boundary_flux("left"),
        bc_right_flux=bump_boundary_flux("right"))
    fields = swe_1d.reconstruct_fields(coeff, zb_faces)
    h = fields[0]
    h_avg = h.mean(dim=-1) if h.dim() > 1 else h
    exact = bump_steady_state(x)["h"]
    err = float(torch.linalg.norm(h_avg - exact) / torch.linalg.norm(exact))
    return err, meta


def test_time_marched_steady_state_below_three_percent_and_second_order():
    """Starting at the exact steady state the solver settles on an O(dx^2) one."""
    err50, meta50 = _steady_run(50, 1, 4.0)
    err100, meta100 = _steady_run(100, 1, 4.0)
    assert meta50["reached_final_time"] and meta100["reached_final_time"]
    for err, name in ((err50, "N=50"), (err100, "N=100")):
        assert err < 0.03, f"{name}: relative L2 depth error {err:.3e} >= 3%"
        assert err < 2e-3, f"{name}: relative L2 depth error {err:.3e} >= 0.2%"
    order = math.log(err50 / err100) / math.log(2.0)
    assert 1.85 <= order <= 4.0, f"observed steady-state order {order:.2f}"