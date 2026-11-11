"""Pins for the extrapolated isothermal-wall heat flux (m15258).

The pins mirror the predeclared gates of
``benchmarks/verify_ns_wall_flux_extrapolation.py`` (evidence:
``validation/ns_heat_conduction/wall_flux_extrapolation.json``) and freeze the
mechanism, not just the numbers:

* the Lagrange weights extrapolate exactly (``sum w = 1``, ``sum w s = 0``);
* on the exact steady conduction state the *solver* wall heat flux is order 1 at
  P1 (baseline of the diagnosis report) while the extrapolated one is order 2,
  and its finest-mesh error is < 0.1 %;
* at P2 the extrapolated flux is still order 2 (no regression);
* the estimator is blind to ``wall_gradient`` (it only reads interior faces);
* on a curved temperature profile -- where the heat flux varies across the gap --
  the nearest interior face is only order 1, i.e. the extrapolation *stencil* is
  what carries the accuracy, not the superconvergence of one face;
* the API contract (empty boundary, periodic axis, too-small axis) holds.

Run: ``PYTHONPATH=. python -m pytest -q tests/test_ns_wall_flux_extrapolation.py``
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

import tensordg.ns_dg_2d as ns
from tensordg.wall_flux_extrapolation import (
    extrapolation_weights,
    wall_heat_flux_extrapolated,
)
from benchmarks.verify_ns_wall_flux_extrapolation import (
    BETA,
    DEFAULT_CELLS,
    KAPPA,
    MU,
    NX,
    PRANDTL,
    REFERENCE,
    T_COLD,
    T_HOT,
    _boundary,
    _case,
    observed_order,
)

EVIDENCE = Path("validation/ns_heat_conduction/wall_flux_extrapolation.json")
CELLS = (8, 16, 32, 64)


def _rows(family, degree, gradient):
    return [_case(ny, degree, gradient, family) for ny in CELLS]


def _order(family, degree, gradient, wall, estimator, points=None):
    rows = _rows(family, degree, gradient)
    ref = REFERENCE[family][wall]
    if estimator == "direct":
        values = [row["direct"][wall] for row in rows]
    else:
        values = [row["extrapolated"][str(points)][wall] for row in rows]
    return observed_order([abs(v - ref) for v in values], CELLS)


def test_extrapolation_weights_extrapolate_exactly():
    for points, expected in ((1, [1.0]), (2, [2.0, -1.0]), (3, [3.0, -3.0, 1.0])):
        weights = extrapolation_weights(points)
        assert weights == pytest.approx(expected)
        nodes = [float(index) for index in range(1, points + 1)]
        if points == 1:
            # a single node extrapolates as a constant; exactness needs >= 2
            continue
        # a linear profile q(s) = a + b s must be extrapolated exactly to s = 0
        for a, b in ((0.3, -1.7), (1.0, 0.0), (-2.5, 4.0)):
            extrapolated = sum(w * (a + b * s) for w, s in zip(weights, nodes))
            assert extrapolated == pytest.approx(a)
    with pytest.raises(ValueError):
        extrapolation_weights(0)


def test_p1_conduction_wall_flux_reaches_second_order():
    """The reported P1 wall flux: order 1 for the solver, order 2 extrapolated."""
    for gradient in ("taylor", "face"):
        for wall in ("ymin", "ymax"):
            direct = _order("conduction", 1, gradient, wall, "direct")
            extrapolated = _order("conduction", 1, gradient, wall, "extrapolated", 2)
            assert direct < 1.4, (gradient, wall, direct)
            assert extrapolated >= 1.8, (gradient, wall, extrapolated)


def test_p1_conduction_error_below_tenth_of_a_percent():
    rows = _rows("conduction", 1, "taylor")
    ref = REFERENCE["conduction"]["ymin"]
    got = rows[-1]["extrapolated"]["2"]["ymin"]
    assert abs(got - ref) / ref * 100.0 <= 0.1
    # and it beats the solver flux of the same state by a wide margin
    direct = rows[-1]["direct"]["ymin"]
    assert abs(got - ref) < abs(direct - ref) / 20.0


def test_p2_does_not_regress():
    for gradient in ("taylor", "face"):
        for wall in ("ymin", "ymax"):
            extrapolated = _order("conduction", 2, gradient, wall, "extrapolated", 2)
            assert extrapolated >= 1.8, (gradient, wall, extrapolated)


def test_estimator_is_independent_of_the_wall_gradient():
    coeff_case = {ny: _case(ny, 1, "taylor", "conduction") for ny in CELLS}
    for ny in CELLS:
        left = coeff_case[ny]["extrapolated"]["2"]
        right = _case(ny, 1, "face", "conduction")["extrapolated"]["2"]
        for wall in ("ymin", "ymax"):
            assert abs(left[wall] - right[wall]) <= 1e-12


def test_extrapolation_stencil_is_required_for_a_curved_profile():
    """With a varying heat flux the nearest interior face is only first order."""
    for gradient in ("taylor", "face"):
        for wall in ("ymin", "ymax"):
            nearest = _order("curved", 1, gradient, wall, "extrapolated", 1)
            extrapolated = _order("curved", 1, gradient, wall, "extrapolated", 2)
            assert nearest <= 1.3, (gradient, wall, nearest)
            assert extrapolated >= 1.8, (gradient, wall, extrapolated)


def test_curved_profile_finest_mesh_error():
    for wall in ("ymin", "ymax"):
        rows = _rows("curved", 1, "face")
        ref = REFERENCE["curved"][wall]
        got = rows[-1]["extrapolated"]["2"][wall]
        assert abs(got - ref) / ref * 100.0 <= 0.1


def test_api_contract():
    ny = 8
    nodes, weights, basis, derivative, signs = ns.legendre_operators(
        1, 3, torch.device("cpu"), torch.float64)
    dx = dy = 1.0 / ny
    x = (torch.arange(NX, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    from benchmarks.verify_ns_wall_flux_extrapolation import _conduction_state
    coeff = ns._initial_coefficients(_conduction_state, x, y, dx, dy, nodes, weights,
                                     basis, torch.device("cpu"), torch.float64)
    penalty = 2.0

    # no boundary at all -> no wall fluxes
    assert wall_heat_flux_extrapolated(coeff, basis, derivative, signs, weights, dx, dy,
                                       {}, MU, KAPPA, penalty) == {}
    # only periodic axes -> skipped
    assert wall_heat_flux_extrapolated(coeff, basis, derivative, signs, weights, dx, dy,
                                       _boundary(), MU, KAPPA, penalty,
                                       periodic=(True, True)) == {}
    # stencil would leave the domain
    with pytest.raises(ValueError):
        wall_heat_flux_extrapolated(coeff, basis, derivative, signs, weights, dx, dy,
                                    _boundary(), MU, KAPPA, penalty, points=ny)
    # per-wall-cell tensors, one value per tangential cell
    flux = wall_heat_flux_extrapolated(coeff, basis, derivative, signs, weights, dx, dy,
                                       _boundary(), MU, KAPPA, penalty)
    assert set(flux) == {"ymin", "ymax"}
    for value in flux.values():
        assert value.shape == (NX,) and torch.isfinite(value).all()


def test_evidence_file_matches_the_capability():
    payload = json.loads(EVIDENCE.read_text())
    assert payload["decision_gates_pass"] is True
    rows = {(row["family"], row["degree"], row["wall_gradient"], row["wall"],
             row["estimator"]): row for row in payload["benchmark"]}
    for gradient in ("taylor", "face"):
        for wall in ("ymin", "ymax"):
            row = rows[("conduction", 1, gradient, wall, "extrapolated_points2")]
            assert row["observed_order"] >= 1.8
            assert abs(row["rel_err_pct"]) <= 0.1
            direct = rows[("conduction", 1, gradient, wall, "direct")]
            assert direct["observed_order"] < 1.4
    assert payload["meta"]["mu"] == pytest.approx(MU)
    assert payload["meta"]["kappa"] == pytest.approx(KAPPA)
    assert payload["meta"]["prandtl"] == pytest.approx(PRANDTL)
    assert payload["meta"]["beta"] == pytest.approx(BETA)
    assert payload["meta"]["cells"] == list(DEFAULT_CELLS)
    assert payload["meta"]["temperatures"] == {"hot": T_HOT, "cold": T_COLD}
    ref = payload["meta"]["reference_wall_flux"]["conduction"]["ymin"]
    assert math.isclose(ref, KAPPA * (T_HOT - T_COLD), rel_tol=1e-15)