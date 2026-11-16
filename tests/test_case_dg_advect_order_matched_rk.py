"""Acceptance guards for the CASE-DG-ADVECT-ORDER matched-RK deliverable.

The round-2 bundle (``benchmarks/case_dg_advect_order_matched_rk.py`` ->
``validation/advection_order_matched_rk/metrics.json`` ->
``docs/CASE-DG-ADVECT-ORDER-round2.md``) rests on two properties that nothing
asserted:

1. **The stage-matched ladder is what it claims.**  ``tableau_for_degree(p)``
   is chosen by *stage count* (``p + 1``) and must therefore report an order
   that an explicit method with that many stages can actually reach (Butcher
   barrier: 5 stages cannot be 5th order, which is why the ``p = 4`` rung runs
   ``ssp_rk54`` = 5 stages / 4th order and is certified as ``RK5DG4``).  A
   regression that silently promised order ``p + 1`` at ``p = 4`` would make the
   accuracy study claim a rate the scheme cannot deliver.

2. **The two RK code paths are one scheme.**  ``solve_advection`` keeps
   hand-coded kernels for ``"rk4"``/``"ssp_rk3"`` for bit-for-bit continuity
   while every other integrator goes through ``runge_kutta.rk_step`` and a
   registered tableau.  Round 2 recorded the missing proof as an open item
   ("局限 4"); ``benchmarks/advorder_rk_path_equivalence.py`` now measures it and
   ``docs/CASE-DG-ADVECT-ORDER-round2-limitation4.md`` carries the numbers.  The
   tests here pin the equality itself, the metadata single-source rule, and the
   fact that the committed note is the renderer's output for the committed
   evidence (no hand editing, no stale verdict).

Numbers in prose that cannot be regenerated (the ``E_time / E_total`` shares
inside ``runge_kutta.tableau_for_degree``'s docstring) are checked against
``metrics.json`` so the prose cannot drift away from the evidence either.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
METRICS = ROOT / "validation/advection_order_matched_rk/metrics.json"
EVID = ROOT / "validation/advection_order_matched_rk/path_equivalence.json"
NOTE = ROOT / "docs/CASE-DG-ADVECT-ORDER-round2-limitation4.md"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensordg.advection_1d as adv                                       # noqa: E402
from tensordg.runge_kutta import (                                        # noqa: E402
    MAX_ORDER_FOR_STAGES, TABLEAUS, max_order_for_stages, tableau_for_degree,
)

#: float64 round-off for the small 1-D configurations used here.
ROUNDOFF_TOL = 1e-12
HAND_CODED = ("rk4", "ssp_rk3")
GENERIC = ("euler", "rk2", "rk3", "ssp_rk54", "rk5")


def _solve(name: str, generic: bool, **kwargs):
    """``solve_advection`` with the hand-coded kernel optionally bypassed."""
    saved = adv._GENERIC_TABLEAUS
    if generic:
        adv._GENERIC_TABLEAUS = (*saved, name)
    try:
        _, coeff, meta = adv.solve_advection(time_integrator=name, **kwargs)
    finally:
        adv._GENERIC_TABLEAUS = saved
    return coeff, meta


def _load_benchmark_module():
    """Import ``benchmarks/advorder_rk_path_equivalence.py`` as a module."""
    path = ROOT / "benchmarks/advorder_rk_path_equivalence.py"
    spec = importlib.util.spec_from_file_location("advorder_rk_path_equivalence", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# 1. matched-stage ladder: order, stages and the Butcher barrier
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("degree", range(0, 9))
def test_tableau_for_degree_honours_its_own_contract(degree):
    tab = tableau_for_degree(degree)
    assert tab.order <= max_order_for_stages(tab.stages), (
        f"p={degree}: {tab.name} claims order {tab.order} with {tab.stages} "
        f"stages, above the explicit-RK barrier {max_order_for_stages(tab.stages)}")
    assert tab.order <= tab.stages
    # the selection rule is stage count first: p + 1 stages whenever a
    # registered tableau has exactly that many stages
    want = [t for t in TABLEAUS.values() if t.stages == degree + 1]
    if want:
        assert tab.stages == degree + 1, (degree, tab)
        assert tab.order == max(t.order for t in want), (degree, tab)


def test_p4_rung_is_five_stage_fourth_order():
    """The RKDG column is called RK5DG4: 5 stages, 4th order -- not order 5."""
    tab = tableau_for_degree(4)
    assert (tab.name, tab.stages, tab.order) == ("ssp_rk54", 5, 4)
    assert tab.order == min(5, max_order_for_stages(5)) == 4


def test_butcher_barrier_table_is_not_optimistic():
    for stages, order in MAX_ORDER_FOR_STAGES.items():
        assert order <= stages
    # the one entry the accuracy study depends on
    assert MAX_ORDER_FOR_STAGES[5] == 4


# --------------------------------------------------------------------------- #
# 2. the two RK paths are one scheme
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,degree", [("rk4", 0), ("rk4", 2),
                                         ("ssp_rk3", 1), ("ssp_rk3", 3)])
def test_unlimited_hand_coded_kernel_equals_registered_tableau(name, degree):
    """Bit-wise equivalence of the hand-coded kernel and ``rk_step``.

    This is the assertion round 2 declared missing (局限 4): same call, same
    arguments, only the driving path differs.
    """
    kw = dict(ncells=12, degree=degree, final_time=0.2, cfl=0.1)
    a, meta_a = _solve(name, False, **kw)
    b, meta_b = _solve(name, True, **kw)
    assert (meta_a["steps"], meta_a["dt"]) == (meta_b["steps"], meta_b["dt"])
    scale = float(a.abs().max())
    rel = float((a - b).abs().max()) / scale
    assert rel <= ROUNDOFF_TOL, (name, degree, rel)


@pytest.mark.parametrize("name", HAND_CODED + GENERIC)
def test_reported_order_comes_from_the_registered_tableau(name):
    """No second copy of the scheme's order/stage count in the solver."""
    _, _, meta = adv.solve_advection(ncells=8, degree=1, final_time=0.05, cfl=0.05,
                                     time_integrator=name)
    tab = TABLEAUS[name]
    assert (meta["rk_order"], meta["rk_stages"]) == (tab.order, tab.stages)
    assert meta["time_integrator"] == name


def test_nonlinear_limiter_is_why_the_two_paths_stay_separate():
    """WENO/AO limiters couple to the recombination -> paths differ measurably.

    Documented, measured and bounded; if this ever became zero the two paths
    could simply be merged and the note in ``docs/`` would be wrong.
    """
    def square(x):
        return torch.where((x % 1.0) < 0.5, torch.ones_like(x), -torch.ones_like(x))

    kw = dict(ncells=10, degree=2, final_time=0.2, cfl=0.1,
              initial=square, limiter="weno")
    a, _ = _solve("ssp_rk3", False, **kw)
    b, _ = _solve("ssp_rk3", True, **kw)
    rel = float((a - b).abs().max()) / float(a.abs().max())
    assert ROUNDOFF_TOL < rel <= 1e-3, rel


# --------------------------------------------------------------------------- #
# 3. the committed evidence and the note cannot drift
# --------------------------------------------------------------------------- #
def test_limitation4_note_is_the_rendered_evidence():
    mod = _load_benchmark_module()
    m = json.loads(EVID.read_text(encoding="utf-8"))
    assert mod.render(m) == NOTE.read_text(encoding="utf-8"), (
        "docs/CASE-DG-ADVECT-ORDER-round2-limitation4.md is not the renderer "
        "output for validation/advection_order_matched_rk/path_equivalence.json "
        "-- regenerate it with `PYTHONPATH=. python "
        "benchmarks/advorder_rk_path_equivalence.py`")


def test_limitation4_evidence_keeps_its_predeclared_thresholds():
    m = json.loads(EVID.read_text(encoding="utf-8"))
    mod = _load_benchmark_module()
    assert m["thresholds"]["ROUNDOFF_TOL"] == mod.ROUNDOFF_TOL
    assert m["thresholds"]["LIMITER_BOUND"] == mod.LIMITER_BOUND
    # every recorded row must still satisfy the checks it was judged by
    for row in m["rows"]:
        cap = (mod.ROUNDOFF_TOL if row["case"] == "unlimited"
               else mod.LIMITER_BOUND)
        assert row["rel_diff"] <= cap, row
    assert m["decision"] in ("merge", "keep")
    assert m["verdict"] == ("PASS" if not m["failed"] else "FAIL")


def test_round2_evidence_bundle_is_still_pass():
    """The #672 acceptance bundle itself: all five checks, no unrun items."""
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    assert m["verdict"] == "PASS" and m["failed"] == []
    assert all(m["checks"].values()), m["checks"]
    for probe in m["dt_order_probes"]:
        assert abs(probe["fitted_time_order"] - probe["design_order"]) <= 0.2, probe


def test_documented_time_shares_match_the_measured_evidence():
    """The E_time/E_total shares quoted in the solver docstring are measured."""
    doc = tableau_for_degree.__doc__ or ""
    quoted = re.search(r"of\s+([\d.]+)\s*%\s*/\s*([\d.]+)\s*%\s*/\s*([\d.]+)\s*%\s*/\s*"
                       r"([\d.]+)\s*%", doc)
    assert quoted, "tableau_for_degree docstring lost its measured shares"
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    worst: dict[int, float] = {}
    for row in m["error_split"]:
        worst[row["degree"]] = max(worst.get(row["degree"], 0.0), row["time_share"])
    for degree, text in zip((1, 2, 3, 4), quoted.groups()):
        measured = 100.0 * worst[degree]
        assert abs(float(text) - measured) <= 0.05 * measured + 0.005, (
            f"docstring says {text} % for p={degree}, evidence says {measured:.4g} %")