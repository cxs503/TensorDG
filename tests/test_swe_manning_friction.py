"""Manning bottom friction in ``tensordg/swe_1d.py``: gates and gate power.

Every test here is a *falsifiable* claim about the drag source; the reference is
the document ``docs/swe_manning_friction.md`` and the artifact
``validation/swe_manning/metrics.json`` produced by
``validation/swe_manning_friction/study.py``.

The suite deliberately keeps its own runs small (a few seconds each) and treats
the full-size study as the shipped artifact: the tests recompute the physics on
coarse grids and, separately, check that the artifact and the document block
still agree with each other and with their gates.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import warnings

import pytest
import torch

from tensordg.swe_1d import (GRAVITY, manning_friction_slope, manning_normal_depth,
                             solve_swe_1d)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY_PATH = os.path.join(ROOT, "validation", "swe_manning_friction", "study.py")
METRICS = os.path.join(ROOT, "validation", "swe_manning", "metrics.json")
DOC = os.path.join(ROOT, "docs", "swe_manning_friction.md")

FLAT_IC = lambda xx: torch.full_like(xx, 1.0)          # noqa: E731
ZERO = lambda xx: torch.zeros_like(xx)                 # noqa: E731

FIXPOINT_GATE = 1.0e-9
GATE_POWER = 1.0e-4


def _study():
    spec = importlib.util.spec_from_file_location("swe_manning_study", STUDY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# 1. analytic references, recomputed on a coarse grid
# --------------------------------------------------------------------------- #
def test_flat_bed_decay_matches_exact_drag_ode():
    """Uniform flow over a flat bed: du/dt = -g n² u²/h^(4/3) has an exact solution.

    The state is uniform, so the DG flux divergence vanishes; the scheme must
    therefore reproduce the ODE to round-off.  A sign error, a missing h or a
    wrongly scaled n all move the error by many orders of magnitude.
    """
    n, h0, u0, t_end = 0.03, 1.0, 1.0, 5.0
    _x, coeff, _zb, meta = solve_swe_1d(
        FLAT_IC, lambda xx: torch.full_like(xx, u0), ZERO,
        ncells=4, degree=1, final_time=t_end, domain=(0.0, 8.0), cfl=0.4,
        periodic=True, manning_n=n)
    h, hu = coeff[:, 0, 0], coeff[:, 0, 1]
    u_num = float((hu / h).mean())
    u_exact = u0 / (1.0 + (GRAVITY * n ** 2 / h0 ** (4.0 / 3.0)) * u0 * t_end)
    assert meta["reached_final_time"]
    assert abs(u_num - u_exact) / u_exact < 1e-9, (u_num, u_exact)
    # uniform flow stays uniform: no higher modes, no depth drift
    assert float((h - h0).abs().max()) < 1e-12
    assert float(coeff[:, 1:, :].abs().max()) < 1e-12
    # the drag limiter must not be what produces this result
    assert meta["friction_capped_steps"] == 0


def test_manning_helpers_are_consistent_with_the_momentum_balance():
    """h_n must solve g h S0 = g n² u² / h^(1/3) with u = q/h (to round-off)."""
    q, slope, n = 1.0, 1e-3, 0.03
    h_n = manning_normal_depth(q, slope, n)
    u_n = q / h_n
    drive = GRAVITY * h_n * slope
    sink = GRAVITY * n ** 2 * u_n ** 2 / h_n ** (1.0 / 3.0)
    assert abs(drive - sink) / drive < 1e-14
    # Manning's velocity form u = (1/n) h^(2/3) √S0
    assert abs(u_n - (1.0 / n) * h_n ** (2.0 / 3.0) * math.sqrt(slope)) < 1e-14
    # friction slope of the uniform state equals the bed slope
    s_f = float(manning_friction_slope(torch.tensor(u_n, dtype=torch.float64),
                                       torch.tensor(h_n, dtype=torch.float64), n))
    assert abs(s_f - slope) / slope < 1e-12
    with pytest.raises(ValueError):
        manning_normal_depth(-1.0, slope, n)
    with pytest.raises(ValueError):
        manning_normal_depth(q, slope, 0.0)


def test_normal_depth_is_a_discrete_fixed_point_and_the_gate_has_power():
    """Slope source and friction sink must cancel to round-off at h_n.

    Gate power is probed twice on the *same* reference state: with the sink
    switched off (n_run = 0) and with a 1 % roughness error, the state must move
    by more than 1e-4, i.e. the gate is able to fail.
    """
    study = _study()
    nominal = study.normal_depth(t_end=8.0, ncells=4)
    assert nominal["reached_final_time"]
    assert nominal["rel_state_drift"] < FIXPOINT_GATE, nominal["rel_state_drift"]
    assert nominal["friction_capped_steps"] == 0

    for probe in (study.normal_depth(t_end=8.0, ncells=4, n_run=0.0),
                  study.normal_depth(t_end=8.0, ncells=4, n_run=1.01 * 0.03)):
        assert probe["rel_state_drift"] > GATE_POWER, probe["rel_state_drift"]


# --------------------------------------------------------------------------- #
# 2. dry-bed behaviour and the drag limiter
# --------------------------------------------------------------------------- #
def test_dry_bed_dam_break_with_friction_is_physical_and_conserves_mass():
    """Friction must not break the wetting front: finite, h >= 0, mass kept."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # the front legitimately warns nothing
        x, coeff, _zb, meta = solve_swe_1d(
            lambda xx: torch.where(xx < 0.0, torch.ones_like(xx), torch.zeros_like(xx)),
            ZERO, ZERO, ncells=24, degree=1, final_time=0.3, domain=(-3.0, 3.0),
            cfl=0.3, manning_n=0.03)
    assert meta["reached_final_time"]
    h = coeff[:, 0, 0]
    assert torch.isfinite(coeff).all()
    assert float(h.min()) >= 0.0
    mass = float(h.sum() * (6.0 / 24))
    assert abs(mass - 3.0) / 3.0 < 1e-5, mass


def test_friction_shortens_runout_monotonically():
    """More roughness, less runout -- the direction of the sink, not its size.

    Resolution caveat: on the 24-cell grid (dx = 0.25 m) the runout is only
    resolved to ±1 cell, so ``n = 0`` and ``n = 0.03`` can land in the same cell
    (measured: 1.375/1.375/1.125 m).  The strict three-way ordering is a claim
    about the *shipped* grid and is gated there, from the artifact
    (``case_c_runout_monotone``, 60 cells: 2.450/2.050/1.750 m); here the
    physics that must hold at any resolution is asserted: non-increasing runout
    and a strict drop for the strong-friction pair.
    """
    def front(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x, coeff, _zb, _meta = solve_swe_1d(
                lambda xx: torch.where(xx < 0.0, torch.ones_like(xx),
                                       torch.zeros_like(xx)),
                ZERO, ZERO, ncells=24, degree=1, final_time=0.3, domain=(-3.0, 3.0),
                cfl=0.3, manning_n=n)
        assert torch.isfinite(coeff).all()
        assert float(coeff[:, 0, 0].min()) >= 0.0
        return float(x[coeff[:, 0, 0] > 1e-3].max())

    f0, f1, f2 = front(0.0), front(0.03), front(0.06)
    assert f0 >= f1 > f2, (f0, f1, f2)


def test_zero_friction_is_bitwise_identical_to_the_legacy_scheme():
    study = _study()
    result = study.zero_friction_is_a_noop()
    assert result["bitwise_identical"] and result["max_abs_difference"] == 0.0


def test_negative_manning_number_is_rejected():
    with pytest.raises(ValueError):
        solve_swe_1d(FLAT_IC, lambda xx: torch.zeros_like(xx), ZERO,
                     ncells=4, final_time=0.01, manning_n=-0.01)


def test_unbounded_drag_switch_warns():
    """friction_limiter=False is an audit switch; it must announce itself."""
    with pytest.warns(RuntimeWarning):
        solve_swe_1d(FLAT_IC, lambda xx: torch.full_like(xx, 1.0), ZERO,
                     ncells=4, final_time=0.01, domain=(0.0, 1.0),
                     manning_n=0.03, friction_limiter=False)


# --------------------------------------------------------------------------- #
# 3. the shipped artifact and the document stay bound to each other
# --------------------------------------------------------------------------- #
def test_artifact_exists_with_provenance_and_all_gates_pass():
    assert os.path.exists(METRICS), (
        "run `PYTHONPATH=. python validation/swe_manning_friction/study.py`")
    metrics = json.load(open(METRICS))
    provenance = metrics["provenance"]
    assert provenance["generator"].endswith("validation/swe_manning_friction/study.py")
    assert provenance["git_commit"] and provenance["torch"]
    gates = metrics["gates"]
    for key in ("case_a_pass", "case_b_pass", "case_b_gate_power_pass",
                "case_c_pass", "case_d_pass"):
        assert gates[key] is True, (key, gates)
    # the numbers quoted by the document must be the artifact's numbers
    assert gates["case_a_max_rel_error"] < gates["case_a_gate"]
    assert gates["case_b_rel_state_drift"] < gates["case_b_gate"]
    assert gates["case_b_gate_power_min_probe_drift"] > gates["case_b_gate_power_gate"]
    assert gates["case_c_runout_monotone"] is True


def test_document_block_is_generated_from_the_artifact():
    """Anti-rot: the results block in the document == regenerated from metrics.json."""
    study = _study()
    metrics = json.load(open(METRICS))
    text = open(DOC).read()
    assert study.BEGIN in text and study.END in text
    block = text.split(study.BEGIN, 1)[1].split(study.END, 1)[0].strip()
    assert block == study.results_block(metrics).strip(), (
        "docs/swe_manning_friction.md §3 was hand-edited or the artifact is stale; "
        "re-run validation/swe_manning_friction/study.py")