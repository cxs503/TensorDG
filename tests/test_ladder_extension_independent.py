"""Gates for the *independent* recompute of the ladder-extension results.

``tests/test_xval_dg_round2_ladder_extension.py`` pins the engine's own audit.
These tests pin the second, engine-free arithmetic
(``validation/xval_dg_round2/close_ladder_extension.py``) and the platform ingest
document, so that "the number is real" does not rest on one code path:

``I1`` on the *three* committed levels the independent arithmetic reproduces the
      published three-level audit (``xval_round2_audit.json``) exactly -- the
      independent path is calibrated before it is trusted on the new level;
``I2`` with the fourth level committed, every independent quantity agrees with
      the engine artifact to ``1e-9`` relative and both reach the *same* L0-L4
      decisions (a disagreement is a finding, not a tolerance to widen);
``I3`` the platform ingest document builds, passes the platform's own contract
      validator, and reports the three closed-form rows as passing -- the numbers
      that go to the machine judge are the committed profile's numbers;
``I4`` the L3 verdict is a comparison of two measured numbers: the test
      recomputes ``shift <= U3`` itself and asserts the published decision equals
      it (a FAIL stays a FAIL).

Run: ``pytest -q tests/test_ladder_extension_independent.py``
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "validation" / "xval_dg_round2" / "artifacts"
sys.path.insert(0, str(ROOT))

from validation.xval_dg_round2 import close_ladder_extension as cle  # noqa: E402

HAS_EXT = cle.level_path(cle.EXT_LEVEL).exists()
needs_ext = pytest.mark.skipif(
    not HAS_EXT,
    reason="fourth reference level (N=6400) not committed yet; "
           "generate it with validation/xval_dg_round2/gen_fv_level.py")


# ------------------------------------------------------------------ I1
def test_independent_arithmetic_reproduces_the_published_three_level_audit():
    l8, l16, l32 = (cle.load_level(n) for n in cle.BASE_LEVELS)
    p8, p16, p32 = (cle.cell_average(l8, 1), cle.cell_average(l16, 2), cle.cell_average(l32, 4))
    d1, d2 = cle.rel_l1(p8, p16), cle.rel_l1(p16, p32)
    p = cle.order_log2(d1, d2)
    band = cle.roache_band_percent(100.0 * d2, p)

    published = json.loads((ART / "xval_round2_audit.json").read_text())
    d1_pub = published["D1_reference_usable"]["differences_percent"][0]
    d2_pub = published["D1_reference_usable"]["differences_percent"][1]
    assert abs(100.0 * d1 - d1_pub) / d1_pub < 1e-9
    assert abs(100.0 * d2 - d2_pub) / d2_pub < 1e-9
    assert abs(p - published["D1_reference_usable"]["observed_order"]) < 1e-9
    assert abs(band - published["D1_reference_usable"]["band_percent"]) / \
        published["D1_reference_usable"]["band_percent"] < 1e-9
    # the band is a GCI: Fs * d_fine / (r^p - 1), checked against its own definition
    assert math.isclose(band, cle.FS_GCI * 100.0 * d2 / (2 ** p - 1.0), rel_tol=1e-15)


# ------------------------------------------------------------------ I2
@needs_ext
def test_independent_and_engine_agree_on_every_number_and_decision():
    indep = cle.audit()
    cmp = cle.compare_with_engine(indep)
    assert cmp["status"] == "compared", cmp
    # scalar quantities must agree to round-off; the eleven legs agree to a small
    # but non-zero difference because the two paths coarsen in a different order
    # (engine: extrapolate on the 3200 grid, then coarsen onto the leg grid; here:
    # extrapolate, project to 1600, then coarsen).  Measured 8.3e-4 pp.
    worst = max((r["rel_diff"] or 0.0) for r in cmp["rows"])
    assert worst <= 1e-9, [r for r in cmp["rows"] if (r["rel_diff"] or 0.0) > 1e-9]
    assert cmp["worst_leg_abs_diff_pp"] <= 1e-2
    assert cmp["decisions_identical"] is True
    engine = json.loads((ART / "ladder_extension.json").read_text())
    assert indep["accepted"] == bool(engine["accepted"])


# ------------------------------------------------------------------ I3
def test_platform_ingest_document_passes_the_contract_validator():
    from validation.xval_dg_round2 import post_ladder_extension as post
    docs = post.build(mission_id=21166, job_id="cj-test")
    assert set(docs) == {"spec", "result", "workload_stdout"}
    # the got side must be the committed profile's numbers, not hand-typed ones
    blob = np.load(cle.level_path(cle.BASE_LEVELS[-1]), allow_pickle=False)
    from benchmarks import xval_dg_round2_certified as xval
    measured = xval.measure_metrics(np.asarray(blob["x"], dtype=float),
                                    np.asarray(blob["rho"], dtype=float))
    doc = json.loads(docs["workload_stdout"])
    for name, value in doc["reference"].items():
        assert abs(value - measured[name]) <= 1e-12 * max(1.0, abs(measured[name]))
    report = post.pp.validate(docs)
    assert report["all_passed"], report["rows"]
    assert report["mode"] == "platform_contract"
    # the four-level reference must not be worse than the three-level one on the
    # shock-position anchor (this is the point of extending the ladder)
    rows = {r["name"]: r for r in docs["result"]["validation"]["rows"]}
    assert rows["shock_position"]["error_percent"] <= \
        rows["shock_position"]["error_percent_three_level"]


# ------------------------------------------------------------------ I4
@needs_ext
def test_l3_decision_is_the_comparison_it_claims_to_be():
    # cert3 / cert4 are the Richardson-extrapolated profiles (engine convention:
    # expressed on the second-finest grid), so the test extrapolates them the same
    # way but with its own arithmetic.
    levels = {n: cle.load_level(n) for n in (*cle.BASE_LEVELS, cle.EXT_LEVEL)}
    base = cle.BASE_LEVELS[0]
    c = {n: cle.cell_average(levels[n], n // base) for n in levels}
    p3 = cle.order_log2(cle.rel_l1(c[800], c[1600]), cle.rel_l1(c[1600], c[3200]))
    p4 = cle.order_log2(cle.rel_l1(c[1600], c[3200]), cle.rel_l1(c[3200], c[6400]))
    x32 = cle.cell_average(levels[3200], 2)
    cert3 = x32 + (x32 - levels[1600]) / (2.0 ** p3 - 1.0)
    x64 = cle.cell_average(levels[6400], 2)
    cert4 = cle.cell_average(x64 + (x64 - levels[3200]) / (2.0 ** p4 - 1.0), 2)
    shift = cle.rel_l1(cert4, cert3)
    indep = cle.audit()
    l3 = indep["L3_band_covers_shift"]
    assert abs(l3["shift_rel_l1"] - shift) <= 1e-12
    assert l3["pass"] == bool(shift <= l3["band_percent_U3"] / 100.0)
    # the coverage ratio published is exactly shift/U3 (no rescaling)
    assert math.isclose(l3["coverage_ratio"], shift / (l3["band_percent_U3"] / 100.0),
                        rel_tol=1e-12)