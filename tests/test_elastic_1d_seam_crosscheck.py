"""Past-seam (``c t > L/2``) cross-check of the d'Alembert elastic-rod case.

Case-pool anchor: ``CASE-DG-DALEMBERT-ELASTIC-1D`` (TensorDG).

The committed case grades ``c t = 0.2``; the reflections benchmark grades the
wrapped regime against one closed form (the periodic image series).  These
tests pin the layer that makes the *wrapped* verdicts trustworthy:

* the shipped image series (the reference the reflections benchmark actually
  grades against) agrees, past the seam, with a Fourier exact-in-time
  propagator and with a from-scratch numpy image sum -- so a sign/period error
  in the grading reference cannot go unnoticed;
* the DG solution agrees, past the seam, with a reference it has never met
  (R1) at the same 2 % gate the case pool uses;
* the DG error still falls at the theoretical order ``p+1`` *through* the
  seam, i.e. the wrapped regime is not quietly first order;
* the wrapped regime makes a *new* falsifiable prediction -- the packet leaves
  the probe at ``t = 0.3``, wraps the seam and returns mirrored at
  ``t = (centre - x_p + L)/c`` with the same ``|sigma|`` and opposite sign --
  and the solver reproduces it;
* the unguarded single-image form is refused past the seam, and if it were
  used as a grading reference the deviation would be far above the gate (a
  false failure), so the guard has teeth.
"""
import numpy as np
import pytest
import torch

from benchmarks.elastic_1d_seam_crosscheck import (
    CENTER,
    C,
    LENGTH,
    MIRROR_T,
    PROBE,
    SEAM_T,
    TOL_PCT,
    _local_peak,
    dg_state,
    ref_images_numpy,
    ref_images_shipped,
)
from benchmarks.elastic_1d_crosscheck import probe_series, ref_spectral, ref_yee
from tensordg.cases import elastic_rod_1d as case
from tensordg.elastic_1d import dalembert_pulse, elastic_energy


def _rel_pct(num, ref):
    num, ref = np.asarray(num), np.asarray(ref)
    return float(np.max(np.abs(num - ref)) / np.max(np.abs(ref)) * 100.0)


def _grid(n=601):
    return np.linspace(0.002, 0.998, n)


# --------------------------------------------------------------------------- #
# the grading reference itself must survive the seam
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("t", [0.55, 0.7, 0.9])
def test_shipped_image_reference_matches_the_fourier_propagator(t):
    """R0 (grading reference) vs R1 (modal exact propagator) past the seam."""
    x = _grid()
    assert C * t > SEAM_T
    assert _rel_pct(ref_images_shipped(x, t), ref_spectral(x, t, 1024)) < 1e-6


@pytest.mark.parametrize("t", [0.55, 1.0, 1.3])
def test_independent_numpy_image_sum_matches_the_shipped_one(t):
    """R2 is a from-scratch re-derivation; disagreement would mean a code bug."""
    x = _grid()
    assert _rel_pct(ref_images_shipped(x, t), ref_images_numpy(x, t)) < 1e-9


def test_yee_leapfrog_stays_usable_past_the_seam():
    """R3 belongs to another discretisation family and evolves through the seam."""
    x = _grid()
    # R3 is deliberately a *coarse* independent reference (see
    # docs/CASE-DG-DALEMBERT-ELASTIC-1D-crosscheck.md §5.5: its extrapolated
    # residual sits orders of magnitude above R1).  The gate below is what a
    # discretisation of another family can honestly promise through the seam.
    assert _rel_pct(ref_yee(x, 0.7, ncells=512), ref_spectral(x, 0.7)) < 5e-2


def test_mirror_arrival_is_predicted_by_the_image_reference():
    """The wrapped regime predicts |sigma| equality and a sign flip at t*=0.7."""
    t_first = case.probe_arrival_time(PROBE)
    assert C * MIRROR_T > SEAM_T, "the mirror arrival must be past the seam"
    ref_t = np.linspace(0.0, MIRROR_T, 4001)
    ref_v = ref_images_numpy(np.full_like(ref_t, PROBE), ref_t)
    s1 = float(np.interp(t_first, ref_t, ref_v))
    s2 = float(np.interp(MIRROR_T, ref_t, ref_v))
    assert s1 * s2 < 0.0, "the two arrivals must carry opposite stress sign"
    # the two peaks are equal in closed form; the interp grid costs ~1e-5
    assert abs(abs(s2) / abs(s1) - 1.0) * 100.0 < 1e-2
    # the arrival time is where |sigma| peaks, not just where the formula sits
    grid = np.linspace(MIRROR_T - 0.05, MIRROR_T + 0.05, 1001)
    vals = ref_images_numpy(np.full_like(grid, PROBE), grid)
    assert _local_peak(grid, vals, MIRROR_T) == pytest.approx(MIRROR_T, rel=1e-6)


# --------------------------------------------------------------------------- #
# the solver must agree with a reference it has never met
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("t", [0.7, 0.9])
def test_dg_matches_the_unseen_reference_past_the_seam(t):
    st = dg_state(100, 2, t)
    err = _rel_pct(st["sigma"], ref_spectral(st["x"], t))
    assert err < TOL_PCT, f"DG vs R1 error {err:.3e} % exceeds {TOL_PCT} %"


def test_dg_error_still_falls_at_theoretical_order_through_the_seam():
    """p = 2 must keep O(h^3) past the seam, not degrade to first order."""
    errs = []
    for ncells in (60, 120):
        st = dg_state(ncells, 2, 0.7)
        errs.append(_rel_pct(st["sigma"], ref_spectral(st["x"], 0.7)))
    order = float(np.log2(errs[0] / errs[1]))
    assert order > 2.5, f"observed order {order:.2f} through the seam"


def test_probe_trace_reproduces_the_mirror_arrival():
    """The twin arrival is measured on the shipped stepper, not on the formula."""
    times, values, _meta = probe_series(100, 2, 0.3, PROBE, MIRROR_T + 0.1)
    t1 = _local_peak(times, values, case.probe_arrival_time(PROBE))
    t2 = _local_peak(times, values, MIRROR_T)
    assert abs(t2 - MIRROR_T) / MIRROR_T * 100.0 < 0.05
    assert abs(t1 - case.probe_arrival_time(PROBE)) / t1 * 100.0 < 0.05
    m1 = float(np.interp(t1, times, values))
    m2 = float(np.interp(t2, times, values))
    assert m1 * m2 < 0.0, "DG must reproduce the stress sign flip"
    assert abs(abs(m2) / abs(m1) - 1.0) * 100.0 < 1.0


def test_elastic_energy_drift_across_the_seam_stays_at_the_dissipation_floor():
    """The upwind flux is dissipative by construction: drift must stay tiny.

    The upwind interface flux is dissipative by construction, so this is a
    floor claim, not an exact-conservation claim: the shipped N=200/P2 run
    drifts 2.48e-04 % over two transits, the coarser N=100 grid used here (for
    test speed) ~2.4e-03 %; the gate is 1e-02 %.
    """
    e0 = dg_state(100, 2, 0.0)["energy"]
    for t in (0.7, 1.3):
        e = dg_state(100, 2, t)["energy"]
        assert abs(e - e0) / e0 * 100.0 < 1e-2, f"energy drift at t={t}"


# --------------------------------------------------------------------------- #
# the historical defect: an unguarded wrong reference past the seam
# --------------------------------------------------------------------------- #
def test_single_image_form_refuses_to_be_a_silent_wrong_reference():
    x = torch.as_tensor(_grid(), dtype=torch.float64)
    with pytest.raises(ValueError, match="only exact for"):
        dalembert_pulse(x, 0.7, 1.0, 1.0, CENTER, 0.05, LENGTH)
    _v, s_bad = dalembert_pulse(x, 0.7, 1.0, 1.0, CENTER, 0.05, LENGTH,
                                strict=False)
    # using it as a grading reference would fail a correct solution
    assert _rel_pct(s_bad.numpy(), ref_spectral(_grid(), 0.7)) > TOL_PCT

# --------------------------------------------------------------------------- #
# the degenerate time: c t = L makes the stress vanish identically
# --------------------------------------------------------------------------- #
def test_stress_vanishes_identically_at_one_full_transit():
    """``c t = L`` -> ``V(x+ct) = V(x-ct)`` -> ``sigma == 0``: not a good metric.

    This is the trap that produced a nonsense 1e+12 % relative error when the
    first version of the past-seam benchmark graded ``sigma`` at ``c t = 1``.
    """
    x = _grid(1201)
    assert float(np.max(np.abs(ref_images_numpy(x, 1.0)))) < 1e-12
    assert float(np.max(np.abs(ref_images_shipped(x, 1.0)))) < 1e-12


def test_zero_baseline_anchor_rows_are_not_vacuous():
    """A drift/deviation row with ``ref = 0`` must grade the value, not pass."""
    from benchmarks.elastic_1d_seam_crosscheck import _row

    ok = _row("drift", 0.0, 5e-4, 1e-3)
    bad = _row("drift", 0.0, 5e-2, 1e-3)
    assert ok["verdict"] == "PASS" and bad["verdict"] == "FAIL"
    assert bad["rel_error_pct"] == pytest.approx(5e-2)
