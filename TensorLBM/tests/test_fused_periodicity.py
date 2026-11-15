"""Regression tests for the fused periodic ghost exchange.

Anchor: ``XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED``.

The tests deliberately include *fault-injection* cases: an equality test that
can never fail is worthless, so two of them mutate the exchange plan and
assert that the comparison breaks.
"""

import numpy as np
import pytest

from tensorlbm.fused_periodicity import (
    D2Q9,
    D3Q19,
    D3Q27,
    FusedPeriodicBGK,
    FusedPeriodicStreamer,
    fit_exponential_rate,
    macroscopic,
    mode_amplitude,
    selection_matrix,
    shear_wave,
    taylor_green_2d,
    wrap_plan,
)

STENCILS = [("D2Q9", D2Q9, (6, 8)), ("D3Q19", D3Q19, (5, 6, 4)), ("D3Q27", D3Q27, (4, 5, 3))]


# ---------------------------------------------------------------------------
# plan structure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,stencil,shape", STENCILS)
def test_plan_partitions_every_destination_once(name, stencil, shape):
    plan = wrap_plan(stencil, shape)
    ncell = int(np.prod(shape))
    for q, boxes in plan.items():
        covered = np.zeros(shape, dtype=int)
        for dst, _src, _seam, nelem in boxes:
            covered[dst] += 1
            assert int(np.prod([d.stop - d.start for d in dst])) == nelem
        assert covered.min() == 1 and covered.max() == 1, f"{name} pop {q} not partitioned"


@pytest.mark.parametrize("name,stencil,shape", STENCILS)
def test_selection_matches_bruteforce_wrap(name, stencil, shape):
    """Seam cells must equal a brute-force out-of-domain source enumeration."""
    mask = selection_matrix(stencil, shape)
    brute = np.zeros_like(mask)
    for q, c in enumerate(stencil):
        for idx in np.ndindex(*shape):
            src = [int(i) - int(e) for i, e in zip(idx, c)]
            if any(s < 0 or s >= n for s, n in zip(src, shape)):
                brute[(q,) + idx] = True
    assert np.array_equal(mask, brute)


def test_seam_volume_scales_with_surface_not_volume():
    """Fused selectivity: seam traffic grows like N^(D-1), not N^D."""
    ratios = []
    for n in (8, 16, 32, 64):
        mask = selection_matrix(D2Q9, (n, n))
        ratios.append(mask.sum() / (len(D2Q9) * n * n))
    assert all(a > b for a, b in zip(ratios, ratios[1:])), ratios
    assert ratios[-1] < 0.25 * ratios[0]


# ---------------------------------------------------------------------------
# three back-ends must agree bit-for-bit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,stencil,shape", STENCILS)
def test_streaming_backends_agree_bitwise(name, stencil, shape):
    rng = np.random.default_rng(12345)
    f = rng.standard_normal((len(stencil),) + shape)
    s = FusedPeriodicStreamer(stencil, shape)
    fused = s.stream_fused(f)
    stats_fused = s.stats.as_dict()
    fullcopy = s.stream_fullcopy(f)
    ghost = s.stream_ghost(f)
    assert np.array_equal(fused, fullcopy)
    assert np.array_equal(fused, ghost)
    # the fused pass writes each element once, the full-copy pass twice
    assert stats_fused["n_elem_total"] == len(stencil) * int(np.prod(shape))
    assert stats_fused["n_elem_seam"] == int(selection_matrix(stencil, shape).sum())


@pytest.mark.parametrize("name,stencil,shape", STENCILS)
def test_fault_injection_breaking_the_plan_is_detected(name, stencil, shape):
    """If one seam box is dropped the fused pass must disagree with the copy."""
    rng = np.random.default_rng(7)
    f = rng.standard_normal((len(stencil),) + shape)
    s = FusedPeriodicStreamer(stencil, shape)
    reference = s.stream_fullcopy(f)

    broken = FusedPeriodicStreamer(stencil, shape)
    q = next(i for i in range(len(stencil)) if any(e != 0 for e in stencil[i]))
    boxes = [b for b in broken.plan[q] if b[2]]
    seam_box = boxes[0]
    broken.plan[q] = [b for b in broken.plan[q] if b is not seam_box]
    mutated = broken.stream_fused(f)
    assert not np.array_equal(mutated, reference), "plan mutation went undetected"


def test_fault_injection_wrong_population_index_is_detected():
    """Swapping two populations' plans breaks the copy-equality invariant."""
    shape = (5, 6)
    rng = np.random.default_rng(3)
    f = rng.standard_normal((len(D2Q9),) + shape)
    s = FusedPeriodicStreamer(D2Q9, shape)
    reference = s.stream_fullcopy(f)

    mutated = FusedPeriodicStreamer(D2Q9, shape)
    mutated.plan[1], mutated.plan[2] = mutated.plan[2], mutated.plan[1]
    assert not np.array_equal(mutated.stream_fused(f), reference)


# ---------------------------------------------------------------------------
# exact operator behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,stencil,shape", STENCILS)
def test_delta_wrap_position_is_exact(name, stencil, shape):
    """Advect a single non-zero cell and compare with index arithmetic."""
    s = FusedPeriodicStreamer(stencil, shape)
    q = int(np.argmax(np.abs(stencil).sum(axis=1)))       # a diagonal population
    f = np.zeros((len(stencil),) + shape)
    start = tuple(0 for _ in shape)
    f[(q,) + start] = 1.0
    pos = np.array(start)
    for step in range(3):
        f = s.stream_fused(f)
        pos = (pos + stencil[q]) % np.array(shape)
        got = np.argwhere(f[q] > 0.5)
        assert len(got) == 1
        assert tuple(int(v) for v in got[0]) == tuple(int(v) for v in pos)


def test_mass_is_conserved_by_fused_exchange():
    solver = FusedPeriodicBGK(D2Q9, (24, 24), nu=0.05, mode="fused")
    solver.init_from_velocity(taylor_green_2d((24, 24)))
    m0 = solver.total_mass()
    solver.step(200)
    assert abs(solver.total_mass() - m0) / m0 < 1e-12


def test_net_momentum_of_shear_wave_stays_zero():
    solver = FusedPeriodicBGK(D2Q9, (16, 16), nu=0.05, mode="fused")
    solver.init_from_velocity(shear_wave((16, 16), axis=0, varying=1))
    p0 = solver.momentum().sum(axis=(1, 2))
    solver.step(100)
    p1 = solver.momentum().sum(axis=(1, 2))
    assert np.all(np.abs(p1 - p0) < 1e-10)


# ---------------------------------------------------------------------------
# physics arms
# ---------------------------------------------------------------------------

def test_shear_wave_decay_matches_nu_k2():
    n = 32
    nu = 0.05
    k = 2.0 * np.pi / n
    solver = FusedPeriodicBGK(D2Q9, (n, n), nu=nu, mode="fused")
    u0 = shear_wave((n, n), axis=0, varying=1, amp=0.02)
    solver.init_from_velocity(u0)
    times, amps = [0], [mode_amplitude(solver.f[1] - solver.f[2], (0, 1))]
    for _ in range(20):
        solver.step(25)
        times.append(times[-1] + 25)
        amps.append(mode_amplitude(solver.f[1] - solver.f[2], (0, 1)))
    rate = fit_exponential_rate(times, amps)
    assert abs(rate + nu * k * k) / (nu * k * k) < 0.05, (rate, -nu * k * k)


def test_taylor_green_decay_matches_2_nu_k2():
    n = 32
    nu = 0.05
    k = 2.0 * np.pi / n
    solver = FusedPeriodicBGK(D2Q9, (n, n), nu=nu, mode="fused")
    solver.init_from_velocity(taylor_green_2d((n, n), amp=0.02))
    times, amps = [0], []
    _rho, u = solver.macroscopic()
    amps.append(mode_amplitude(u[0], (1, 1)))
    for _ in range(10):
        solver.step(20)
        times.append(times[-1] + 20)
        _rho, u = solver.macroscopic()
        amps.append(mode_amplitude(u[0], (1, 1)))
    rate = fit_exponential_rate(times, amps)
    analytic = -2.0 * nu * k * k
    assert abs(rate - analytic) / abs(analytic) < 0.05, (rate, analytic)


@pytest.mark.parametrize("mode", ["fused", "fullcopy", "ghost"])
def test_solver_modes_reach_the_same_state(mode):
    states = {}
    for m in ("fused", "fullcopy", "ghost"):
        solver = FusedPeriodicBGK(D3Q19, (6, 6, 6), nu=0.05, mode=m)
        solver.init_from_velocity(shear_wave((6, 6, 6), axis=0, varying=1, amp=0.02))
        solver.step(30)
        states[m] = solver.f.copy()
    ref = states["fused"]
    assert np.allclose(states[mode], ref, rtol=0, atol=0)