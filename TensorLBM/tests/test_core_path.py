"""Quality gate for the T2 unified core path (``tensorlbm.core_path``).

Anchor: ``GAP-LBM-T2-GPU-COREPATH``.  The tests assert *invariants* of the
comparison protocol, not merely that code executes:

1. the fused single-sweep kernel is the **same numerical answer** as the
   un-fused reference kernel (bit-for-bit in float64);
2. the T3 case still passes its pre-registered accuracy gate, so the arms being
   timed are physics-validated;
3. an arm that fails a gate can never be published as a throughput number;
4. a back-end the machine does not have can never silently produce a number;
5. fusion really does reduce live state and keeps state allocation at zero
   during stepping (buffer reuse, not reallocation).
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.core_path import (
    ACCURACY_TOL,
    CONSISTENCY_TOL,
    T3_CASE,
    BackendUnavailable,
    CoreConfig,
    CorePath,
    admissibility,
    probe_backends,
    run_arm,
)


def _cfg(**kw):
    base = dict(stencil="D2Q9", shape=(32, 32), nu=0.05, kernel="fused",
                backend="numpy", dtype="float64")
    base.update(kw)
    return CoreConfig(**base)


# ---------------------------------------------------------------------------
# 1. kernel equivalence: fusion must not change the answer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kernel", ["fused", "fused_slices"])
@pytest.mark.parametrize("dtype,tol", [("float64", 0.0), ("float32", 1e-6)])
def test_fused_is_bitwise_reference(kernel, dtype, tol):
    u0 = 0.01
    k = 2.0 * np.pi / 32
    n = 32
    x = np.arange(n)
    X, Y = np.meshgrid(x, x, indexing="xy")
    u = np.stack([u0 * np.sin(k * X) * np.cos(k * Y),
                  -u0 * np.cos(k * X) * np.sin(k * Y)])

    ref = CorePath(_cfg(kernel="reference", dtype=dtype)).init_from_velocity(u)
    fus = CorePath(_cfg(kernel=kernel, dtype=dtype)).init_from_velocity(u)
    ref.step(7)
    fus.step(7)
    d_ref, d_fus = ref.ops.to_numpy(ref.f), fus.ops.to_numpy(fus.f)
    diff = float(np.max(np.abs(d_ref.astype(np.float64) - d_fus.astype(np.float64))))
    scale = float(np.max(np.abs(d_ref.astype(np.float64))))
    assert diff <= tol, f"fused kernel changed the solution: max|Δ|={diff:.3e}"


@pytest.mark.parametrize("kernel", ["fused", "fused_slices"])
def test_fused_and_reference_velocities_agree(kernel):
    r_ref, u_ref = run_arm(_cfg(kernel="reference"), steps=300, record_every=100,
                           repeat=1, measure_memory=False)
    r_fus, _ = run_arm(_cfg(kernel=kernel), steps=300, record_every=100,
                       repeat=1, measure_memory=False, reference_field=u_ref)
    assert r_fus.consistency_rel_diff is not None
    assert r_fus.consistency_rel_diff <= CONSISTENCY_TOL
    assert r_fus.consistency_pass is True
    assert r_ref.consistency_pass is None      # reference arm: no baseline needed


# ---------------------------------------------------------------------------
# 2. the arms being timed are physics-validated (T3 case, pre-registered gate)
# ---------------------------------------------------------------------------

def test_t3_taylor_green_gate_passes_on_both_kernels():
    for kernel in ("reference", "fused", "fused_slices"):
        res, _ = run_arm(_cfg(kernel=kernel), steps=600, record_every=100,
                         repeat=1, measure_memory=False)
        assert res.accuracy_rel_err <= ACCURACY_TOL, (
            f"{T3_CASE} accuracy gate failed for {kernel}: "
            f"{res.accuracy_rel_err:.3e}")
        assert res.accuracy_pass is True
        assert res.verdict == "ADMISSIBLE"
        assert res.perf_admissible is True
        # fitted rate must be a decay at the analytic rate, not a fudge
        assert res.decay_rate_fit == pytest.approx(res.decay_rate_analytic,
                                                  rel=ACCURACY_TOL)


# ---------------------------------------------------------------------------
# 3. admissibility rules: no physics => no number
# ---------------------------------------------------------------------------

def test_admissibility_rejects_inaccurate_arm():
    v = admissibility(accuracy_rel_err=0.5, consistency_rel_diff=0.0,
                      consistency_tol=CONSISTENCY_TOL)
    assert v["verdict"] == "REJECTED_ACCURACY"
    assert v["perf_admissible"] is False
    assert v["reasons"] and "accuracy" in v["reasons"][0]


def test_admissibility_rejects_inconsistent_arm():
    v = admissibility(accuracy_rel_err=1e-3, consistency_rel_diff=1e-3,
                      consistency_tol=CONSISTENCY_TOL)
    assert v["verdict"] == "REJECTED_CONSISTENCY"
    assert v["perf_admissible"] is False


def test_admissibility_accepts_clean_arm_and_reference_arm():
    assert admissibility(1e-3, 0.0, CONSISTENCY_TOL)["verdict"] == "ADMISSIBLE"
    ref = admissibility(1e-3, None, None)
    assert ref["verdict"] == "ADMISSIBLE"
    assert ref["consistency_pass"] is None


def test_gate_constants_are_the_preregistered_values():
    # tolerance may never exceed the ecosystem acceptance limit (15%)
    assert 0.0 < ACCURACY_TOL <= 0.15
    assert CONSISTENCY_TOL <= 1e-8


# ---------------------------------------------------------------------------
# 4. back-end honesty: unavailable hardware cannot yield a number
# ---------------------------------------------------------------------------

def test_backend_probe_is_total_and_honest():
    probes = probe_backends()
    assert "numpy" in probes and probes["numpy"].available
    for name, p in probes.items():
        assert p.name == name
        if not p.available:
            assert p.reason, f"{name} unavailable without a reason"


def test_unavailable_backend_raises_instead_of_falling_back():
    probes = probe_backends()
    unavailable = [n for n, p in probes.items() if not p.available]
    if not unavailable:
        pytest.skip("this host has every probed back-end")
    with pytest.raises(BackendUnavailable) as exc:
        CorePath(_cfg(backend=unavailable[0]))
    assert probes[unavailable[0]].reason in str(exc.value) or \
        unavailable[0] in str(exc.value)


def test_unknown_backend_raises():
    with pytest.raises(BackendUnavailable):
        CorePath(_cfg(backend="no-such-backend"))


def test_torch_cpu_backend_matches_numpy():
    probes = probe_backends()
    if not probes.get("torch-cpu", None) or not probes["torch-cpu"].available:
        pytest.skip("torch not available")
    n = 24
    k = 2.0 * np.pi / n
    x = np.arange(n)
    X, Y = np.meshgrid(x, x, indexing="xy")
    u = np.stack([0.01 * np.sin(k * X) * np.cos(k * Y),
                  -0.01 * np.cos(k * X) * np.sin(k * Y)])
    a = CorePath(_cfg(shape=(n, n), backend="numpy")).init_from_velocity(u)
    b = CorePath(_cfg(shape=(n, n), backend="torch-cpu")).init_from_velocity(u)
    a.step(5)
    b.step(5)
    ua, ub = a.velocity(), b.velocity()
    assert float(np.max(np.abs(ua - ub))) <= 1e-12


# ---------------------------------------------------------------------------
# 5. memory management: fusion reduces live state, stepping reuses buffers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kernel", ["fused", "fused_slices"])
def test_fused_declares_less_live_memory_than_reference(kernel):
    fused = CorePath(_cfg(kernel=kernel))
    ref = CorePath(_cfg(kernel="reference"))
    assert fused.declared_live_bytes() < ref.declared_live_bytes()
    # exact inventory: 2 QN vs 4 QN state buffers + identical workspace
    n, q = 32 * 32, 9
    item = 8
    assert fused.declared_live_bytes() == (2 * q * n + 5 * n) * item
    assert ref.declared_live_bytes() == (4 * q * n + 5 * n) * item


def test_measured_peak_memory_follows_the_inventory():
    r_ref, _ = run_arm(_cfg(kernel="reference"), steps=200, record_every=100,
                       repeat=1, measure_memory=True)
    r_fus, _ = run_arm(_cfg(kernel="fused"), steps=200, record_every=100,
                       repeat=1, measure_memory=True)
    r_sli, _ = run_arm(_cfg(kernel="fused_slices"), steps=200, record_every=100,
                       repeat=1, measure_memory=True)
    assert 0 < r_fus.measured_peak_bytes < r_ref.measured_peak_bytes
    assert 0 < r_sli.measured_peak_bytes < r_ref.measured_peak_bytes
    # both fused variants declare the same (reduced) live state
    assert r_sli.declared_live_bytes == r_fus.declared_live_bytes


def test_stepping_allocates_no_new_state_buffers():
    for kernel in ("reference", "fused", "fused_slices"):
        solver = CorePath(_cfg(kernel=kernel))
        u = np.zeros((2, 32, 32))
        u[0] = 0.01
        solver.init_from_velocity(u)
        ids = set()
        for _ in range(5):
            solver.step(1)
            ids |= solver.state_buffer_ids()
        assert solver.steady_state_allocations == 0, (
            f"{kernel} allocated new Q*N state inside step()")
        expected = 4 if kernel == "reference" else 2
        assert len(ids) == expected, f"{kernel}: {len(ids)} distinct state buffers"


def test_slice_streaming_eliminates_roll_temporaries():
    """Deterministic proxy for "no full-field temporary per step"."""
    n, q = 24, 9
    u = np.zeros((2, n, n))
    u[0] = 0.01
    counts = {}
    for kernel in ("reference", "fused", "fused_slices"):
        solver = CorePath(_cfg(shape=(n, n), kernel=kernel)).init_from_velocity(u)
        solver.step(3)
        counts[kernel] = solver.roll_calls_per_step
    assert counts["reference"] == q
    assert counts["fused"] == q
    assert counts["fused_slices"] == 0.0


def test_d3q19_shape_is_accepted_by_the_unified_api():
    cfg = CoreConfig(stencil="D3Q19", shape=(8, 8, 8), nu=0.05, kernel="fused")
    solver = CorePath(cfg)
    u = np.zeros((3, 8, 8, 8))
    solver.init_from_velocity(u)
    solver.step(2)
    rho, uu = solver.macroscopic()
    assert np.allclose(rho, 1.0)
    assert np.max(np.abs(uu)) < 1e-14


def test_tg_case_identity_is_the_t3_row():
    assert T3_CASE == "taylor-green-2d"