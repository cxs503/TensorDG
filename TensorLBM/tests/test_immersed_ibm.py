"""Tests for the LIFE-style static immersed-marker obstacle set.

These tests assert *invariants*, not just "the code runs":

1. the delta kernels are partitions of unity (a wrong kernel silently leaks
   momentum, which no Cd comparison would localise);
2. the marker ring follows LIFE's spacing / pruning convention;
3. spread and interpolate are exact discrete adjoints -- this is what makes
   the momentum exchange ``sum_m G_m ds`` conservative;
4. the node force density integrates to the marker traction to round-off;
5. a marker cloud in a quiescent field produces no spurious force;
6. a static marker ring in a uniform stream produces drag in the right
   direction and removes most of the slip;
7. the physical probe runs, stays finite and reports a plausible Cd band plus
   an honest awake/no-wake flag.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.cylinder import run_cylinder, wall_momentum_exchange_x, apply_open_boundaries
from tensorlbm.immersed import (
    StaticMarkerObstacles,
    guo_source,
    kernel_3pt,
    kernel_4pt,
    make_circle_markers,
    multi_direct_forcing,
)
from tensorlbm.lattice import D2Q9


@pytest.mark.parametrize("kernel,offsets", [(kernel_3pt, range(-2, 3)), (kernel_4pt, range(-3, 4))])
def test_kernel_partition_of_unity(kernel, offsets):
    rng = np.random.default_rng(12345)
    xs = np.concatenate([rng.random(2000), np.array([0.0, 0.25, 0.5, 0.75, 0.999])])
    for x in xs:
        s = sum(kernel(j - x) for j in offsets)
        assert abs(s - 1.0) < 1e-12
        m = sum((j - x) * kernel(j - x) for j in offsets)
        assert abs(m) < 1e-12          # first moment: the kernel is centred


def test_circle_markers_follow_life_convention():
    radius, spacing = 8.0, 1.0
    pos, stats = make_circle_markers(20.0, 30.0, radius, spacing=spacing, min_sep=0.5)
    assert stats["n_markers_raw"] == math.floor(2 * math.pi * radius / spacing)
    assert stats["spacing_realised"] == pytest.approx(
        2 * math.pi * radius / stats["n_markers_raw"]
    )
    # equally spaced on the circle: the chord is 2 r sin(pi/n), the arc length
    # carried by each marker (what the IBM uses) is 2 pi r / n
    n = int(stats["n_markers"])
    d = np.linalg.norm(np.diff(np.vstack([pos, pos[:1]]), axis=0), axis=1)
    assert d.max() - d.min() < 1e-9
    chord = 2 * radius * math.sin(math.pi / n)
    assert d.mean() == pytest.approx(chord, rel=1e-9)
    assert stats["spacing_realised"] * n == pytest.approx(2 * math.pi * radius, rel=1e-12)
    # no marker closer than min_sep to another one
    dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    assert dist.min() >= 0.5 - 1e-12


def test_pruning_counts_dropped_markers():
    # dense spacing + the LIFE pruning radius: some markers must be dropped
    pos, stats = make_circle_markers(10.0, 10.0, 6.0, spacing=0.6, min_sep=0.75)
    assert stats["n_pruned"] > 0
    dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    assert dist.min() >= 0.75 - 1e-12


def _markers(shape=(40, 60), radius=5.0, kernel="4pt"):
    pos, stats = make_circle_markers(
        shape[1] / 2.0, shape[0] / 2.0, radius, spacing=1.0, min_sep=0.5
    )
    return StaticMarkerObstacles(shape=shape, positions=pos,
                                 spacing=stats["spacing_realised"], kernel=kernel), pos, stats


def test_weights_are_normalised_for_interior_markers():
    mk, _, _ = _markers()
    ws = mk.weight_summary
    assert ws["max_deviation_from_one"] < 1e-12


def test_spread_and_interpolate_are_adjoints():
    """sum_m G_m * interp(u)_m == sum_x F(x) * u(x)  for random u, G."""
    mk, _, _ = _markers()
    rng = np.random.default_rng(7)
    n_nodes = mk.shape[0] * mk.shape[1]
    u = torch.from_numpy(rng.standard_normal(n_nodes))
    g = torch.from_numpy(rng.standard_normal(mk.n_markers))
    lhs = float((g * mk.interpolate(u)).sum())
    rhs = float((mk.spread(g, n_nodes) * u).sum())
    assert lhs == pytest.approx(rhs, rel=1e-13, abs=1e-12)


def test_node_force_integrates_to_marker_traction():
    mk, _, _ = _markers()
    rng = np.random.default_rng(11)
    gx = torch.from_numpy(rng.standard_normal(mk.n_markers))
    gy = torch.from_numpy(rng.standard_normal(mk.n_markers))
    fx, fy = mk.force_density(gx, gy)
    bx, by = mk.body_force(gx, gy)
    assert float(fx.sum()) == pytest.approx(-bx, rel=1e-13, abs=1e-13)
    assert float(fy.sum()) == pytest.approx(-by, rel=1e-13, abs=1e-13)
    assert bx == pytest.approx(-float(gx.sum()) * mk.spacing, rel=1e-14)


def test_quiescent_field_gives_no_spurious_force():
    mk, _, _ = _markers()
    rho = torch.ones(mk.shape, dtype=torch.float64)
    zero = torch.zeros(mk.shape, dtype=torch.float64)
    gx, gy, info = multi_direct_forcing(mk, rho, zero, zero, u_body=(0.0, 0.0), n_iter=3)
    assert float(gx.abs().max()) < 1e-14
    assert float(gy.abs().max()) < 1e-14
    assert info["slip_before"] == 0.0


def test_uniform_stream_past_static_markers_produces_resistance():
    mk, _, _ = _markers()
    u0 = 0.05
    rho = torch.ones(mk.shape, dtype=torch.float64)
    ux = torch.full(mk.shape, u0, dtype=torch.float64)
    zero = torch.zeros(mk.shape, dtype=torch.float64)
    gx, gy, info = multi_direct_forcing(mk, rho, ux, zero, u_body=(0.0, 0.0), n_iter=3)
    # the obstacle must decelerate the fluid: total traction on the fluid is -x
    assert float(gx.sum()) < 0.0
    assert abs(float(gy.sum())) < 1e-10 * abs(float(gx.sum()))
    # and the slip must actually go down
    assert info["slip_after"] < 0.5 * info["slip_before"]


def test_guo_source_has_zero_moment_and_zero_at_rest():
    rho = torch.ones((6, 7), dtype=torch.float64)
    ux = torch.full((6, 7), 0.05, dtype=torch.float64)
    uy = torch.zeros((6, 7), dtype=torch.float64)
    fx = torch.full((6, 7), 1e-5, dtype=torch.float64)
    fy = torch.zeros((6, 7), dtype=torch.float64)
    src = guo_source(rho, ux, uy, fx, fy, 1.9)
    # no mass source
    assert float(src.sum()) < 1e-15
    # the collision/source pair delivers (1 - omega/2) F of momentum; the
    # remaining F/2 enters through u = (sum f e + F/2)/rho
    mom_x = float((D2Q9.e[:, 0].view(-1, 1, 1) * src).sum())
    assert mom_x == pytest.approx((1.0 - 1.9 / 2.0) * 1e-5 * 6 * 7, rel=1e-12)
    zero = torch.zeros((6, 7), dtype=torch.float64)
    assert float(guo_source(rho, ux, uy, zero, zero, 1.9).abs().max()) == 0.0


def test_wall_momentum_exchange_sign():
    """A wall must remove +x momentum from a +x stream (force on fluid is -x)."""
    ny, nx = 12, 16
    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[0, :] = True
    solid[-1, :] = True
    rho = torch.ones((ny, nx), dtype=torch.float64)
    ux = torch.full((ny, nx), 0.05, dtype=torch.float64)
    uy = torch.zeros((ny, nx), dtype=torch.float64)
    f = D2Q9.equilibrium(rho, ux, uy)
    fw = wall_momentum_exchange_x(f, solid)
    assert fw < 0.0


def test_open_boundaries_impose_inlet_velocity_and_outlet_pressure():
    ny, nx = 12, 20
    rho = torch.ones((ny, nx), dtype=torch.float64)
    ux = torch.full((ny, nx), 0.03, dtype=torch.float64)
    uy = torch.zeros((ny, nx), dtype=torch.float64)
    f = D2Q9.equilibrium(rho, ux, uy)
    apply_open_boundaries(f, 0.05, rho_outlet=1.0)
    r2, ux2, uy2 = D2Q9.macroscopic(f)
    assert float(ux2[1:-1, 0].mean()) == pytest.approx(0.05, rel=2e-2)
    assert float(r2[1:-1, -1].mean()) == pytest.approx(1.0, rel=1e-12)


def test_cylinder_run_is_finite_and_reports_diagnostics():
    """Short physical run: no acceptance claim, only sanity + honest flags."""
    res = run_cylinder(d_cells=10, re=100.0, n_steps=260, average_from=0.5,
                       sample_every=2, threads=2, field_snapshot=False)
    assert np.isfinite(res.cd_mean) and np.isfinite(res.cl_rms)
    assert 0.0 < res.cd_mean < 6.0             # wide band: 10 cells/diameter only
    assert res.rho_min > 0.5 and res.rho_max < 2.0
    assert res.marker_stats["n_markers"] == math.floor(2 * math.pi * (res.d_cells / 2))
    assert "rel_mismatch" in res.momentum_budget
    assert 0.0 <= res.slip_rel_to_u0 < 0.2      # diffuse-IBM slip is reported, not hidden
    assert res.tau > 0.5