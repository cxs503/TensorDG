"""Consistency of the CASE-LAX-DG resolution-closure artifact.

The artifact ``docs/benchmarks/results/lax_resolution_closure/metrics.json`` is
produced by ``benchmarks/lax_resolution_closure.py`` from a long cluster run
(the committed run took 1385 s wall), so it cannot be regenerated in a unit test.  What *can* be checked --
and is what this file checks -- is that the committed numbers still say what
the committed verdict says, i.e. that the verdict is **re-derived** from the
ladder and not asserted next to it:

* the ladder is the registered one (p3, ``200/400/800/1600``, no limiter);
* the reported production leg is the ``N = 400`` entry, bit-identical to it;
* the closure mesh is the smallest *run* mesh whose L2 meets the bar for
  **every** field, or ``None``;
* ``status`` is ``PASS`` exactly when such a mesh exists -- an extrapolated
  ``N*`` never produces a pass (the honest-FAIL rule);
* the limiter arms differ in exactly one knob (``limiter``);
* every field's relative L2 decreases monotonically along the ladder.

Resolution-closure vs ledger bar (m20699)
-----------------------------------------
The standard case pool ladder (P1/P2/P3 × N=100/200/400) yields an A0
verdict of **FAIL** because the production leg (p3.N400) has relative L2
density = 3.116 %, just above the 3 % ledger bar.  The resolution closure
study extends the P3 ladder to N = [200, 400, 800, 1600] and shows that the
3 % threshold is met at N = 800 (worst L2 = 2.315 % for density).  The tests
below machine-verify this finding so it cannot drift from the artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

METRICS = Path("docs/benchmarks/results/lax_resolution_closure/metrics.json")
FIELDS = ("density", "velocity", "pressure")
LADDER = (200, 400, 800, 1600)
ARMS = ("none", "tvb", "weno")
LEDGER_TOLERANCE = 0.03  # 3 % relative L2


@pytest.fixture(scope="module")
def metrics():
    # The artifact lives under ``docs/`` and is tracked by git, so a missing
    # file means the evidence was dropped -- not that the study is optional.
    # Skipping here would turn the whole closure gate green and empty (the
    # m16815 gate-vacuity failure mode), so absence is a hard failure.
    assert METRICS.exists(), (
        f"committed closure artifact {METRICS} is missing; it is tracked under "
        "docs/, so this is a dropped-evidence defect, not a skip"
    )
    data = json.loads(METRICS.read_text(encoding="utf-8"))
    for key in ("config", "ladder", "arms", "gate"):
        assert key in data, f"artifact lacks required key {key!r}"
    return data


def test_ladder_is_the_registered_production_arm(metrics):
    legs = sorted(metrics["ladder"], key=lambda e: e["cells"])
    assert tuple(e["cells"] for e in legs) == LADDER
    assert {e["degree"] for e in legs} == {3}
    assert {e["limiter"] for e in legs} == {"none"}, (
        "the closure ladder must measure the registered unlimited p3 arm; a "
        "limiter here would silently change the acceptance configuration"
    )


def test_production_leg_number_is_the_ladder_entry(metrics):
    gate = metrics["gate"]
    assert gate["production_leg"] == "p3.N400"
    leg = next(e for e in metrics["ladder"] if e["cells"] == 400)
    for field in FIELDS:
        assert gate["production_l2"][field] == leg["l2"][field]


def test_closure_mesh_is_rederived_from_the_ladder(metrics):
    gate = metrics["gate"]
    tol = gate["tolerance"]
    legs = sorted(metrics["ladder"], key=lambda e: e["cells"])
    closing = [e["cells"] for e in legs if all(e["l2"][f] <= tol for f in FIELDS)]
    expected = min(closing) if closing else None
    assert gate["closure_mesh"] == expected
    assert gate["status"] == ("PASS" if expected is not None else "FAIL")


def test_extrapolation_never_counts_as_a_pass(metrics):
    """If no run leg meets the bar, the verdict must stay FAIL."""
    gate = metrics["gate"]
    model = gate["per_field"]["density"]["model"] or {}
    if gate["closure_mesh"] is None:
        assert gate["status"] == "FAIL"
        assert model.get("extrapolated_n") is not None
    else:
        assert gate["closure_mesh"] <= max(LADDER)


def test_every_field_converges_along_the_ladder(metrics):
    legs = sorted(metrics["ladder"], key=lambda e: e["cells"])
    for field in FIELDS:
        series = [e["l2"][field] for e in legs]
        # The ladder must go down overall; a finest-pair plateau is allowed
        # because it is measured (velocity L2 1.383 % at N=800 -> 1.475 % at
        # N=1600), not assumed away.
        assert series[-1] < series[0], f"{field} L2 does not decrease: {series}"
        for coarse, fine in zip(series, series[1:]):
            assert fine <= coarse * 1.15, (
                f"{field} L2 grows by more than 15 % on one refinement step: {series}"
            )


def test_limiter_arms_change_only_the_limiter(metrics):
    arms = metrics["arms"]
    assert tuple(a["limiter"] for a in arms) == ARMS
    assert {a["cells"] for a in arms} == {metrics["config"]["arm_cells"]}
    assert {a["degree"] for a in arms} == {metrics["config"]["degree"]}
    # the unlimited arm at the production mesh must reproduce the ladder entry
    leg = next(e for e in metrics["ladder"] if e["cells"] == metrics["config"]["arm_cells"])
    none_arm = next(a for a in arms if a["limiter"] == "none")
    for field in FIELDS:
        assert none_arm["l2"][field] == leg["l2"][field]


def test_figures_are_declared_on_disk(metrics):
    base = METRICS.parent / "figures"
    if not metrics.get("figures"):
        pytest.skip("figures were skipped for this run")
    for figure in metrics["figures"]:
        assert (base / figure["name"]).exists()
        assert figure["caption"].strip()


# --------------------------------------------------------------- m20699: ledger-bar closure
def _find_leg(metrics: dict, cells: int) -> dict:
    for leg in metrics["ladder"]:
        if leg["cells"] == cells and leg.get("limiter") == "none":
            return leg
    raise KeyError(f"no unlimited leg with cells={cells}")


def test_n800_passes_ledger_bar(metrics):
    """N = 800 at P3 must have all three fields below the 3 % ledger bar.

    This is the key "验证欠账" finding: the ledger bar *is* achievable under
    the registered configuration (P3, unlimited, CFL 0.35); it just requires
    N = 800 rather than N = 400.
    """
    leg = _find_leg(metrics, 800)
    l2 = leg["l2"]
    for field in FIELDS:
        assert l2[field] < LEDGER_TOLERANCE, (
            f"N=800 {field} L2 = {l2[field]:.6f} ({l2[field]*100:.3f}%) "
            f"≥ {LEDGER_TOLERANCE*100:.1f}% ledger bar"
        )


def test_n800_worst_field_is_density(metrics):
    """Density is the hardest field; it should be the worst at every resolution."""
    for cells in LADDER:
        leg = _find_leg(metrics, cells)
        l2 = leg["l2"]
        assert l2["density"] >= l2["velocity"], (
            f"N={cells}: velocity L2 ({l2['velocity']:.6f}) > density L2 ({l2['density']:.6f})"
        )
        assert l2["density"] >= l2["pressure"], (
            f"N={cells}: pressure L2 ({l2['pressure']:.6f}) > density L2 ({l2['density']:.6f})"
        )


def test_n400_still_fails_ledger_bar(metrics):
    """N = 400 at P3 must still fail the 3 % bar (honest FAIL, unchanged).

    This confirms the case-pool artifact's A0 FAIL verdict is correct:
    p3.N400 density L2 = 3.116 % > 3 %.
    """
    leg = _find_leg(metrics, 400)
    l2 = leg["l2"]
    assert l2["density"] >= LEDGER_TOLERANCE, (
        f"N=400 density L2 = {l2['density']:.6f} should be ≥ {LEDGER_TOLERANCE}"
    )


def test_unlimited_arm_is_most_accurate_at_n400(metrics):
    """At N = 400, the unlimited arm must have the lowest L2 density.

    TVB and WENO limiters degrade accuracy because they reduce troubled cells
    to linear reconstruction, which is over-limiting at P3.
    """
    arms = metrics.get("arms", [])
    if not arms:
        pytest.skip("limiter arms not present in closure artifact")

    l2_density = {}
    for arm_data in arms:
        limiter = arm_data.get("limiter", "unknown")
        if "l2" in arm_data and "density" in arm_data["l2"]:
            l2_density[limiter] = arm_data["l2"]["density"]

    if len(l2_density) < 2:
        pytest.skip("insufficient limiter arms for comparison")

    assert "none" in l2_density, "unlimited arm missing from limiter comparison"
    none_l2 = l2_density["none"]
    for arm_name, arm_l2 in l2_density.items():
        if arm_name != "none":
            assert arm_l2 >= none_l2, (
                f"limiter '{arm_name}' L2 ({arm_l2:.6f}) < unlimited ({none_l2:.6f})"
            )


def test_tvb_degrades_accuracy_vs_unlimited(metrics):
    """TVB limiter must degrade L2 density by at least 2× at N = 400."""
    arms = metrics.get("arms", [])
    arm_by_limiter = {a.get("limiter"): a for a in arms}
    if "none" not in arm_by_limiter or "tvb" not in arm_by_limiter:
        pytest.skip("tvb or none arm missing from closure artifact")

    none_l2 = arm_by_limiter["none"]["l2"]["density"]
    tvb_l2 = arm_by_limiter["tvb"]["l2"]["density"]
    ratio = tvb_l2 / none_l2
    assert ratio >= 2.0, (
        f"TVB L2 ({tvb_l2:.6f}) / unlimited L2 ({none_l2:.6f}) = {ratio:.2f} < 2.0"
    )


def test_closure_n800_l2_matches_documentation(metrics):
    """The N=800 L2 density in the artifact must match the published doc number.

    From docs/CASE-LAX-DG-resolution-closure.md: N=800 ρ L2 = 2.315 %.
    """
    leg = _find_leg(metrics, 800)
    expected = 0.02315
    assert abs(leg["l2"]["density"] - expected) < 5e-5, (
        f"N=800 density L2 = {leg['l2']['density']:.6f}, expected ~{expected}"
    )

# ------------------------------------------------- m22300 (reopened #617): summary.md consistency
#
# The committed one-line summary used to be written by the ingest *only when
# it did not exist*, so a summary left by an earlier --quick run ("closure
# mesh 200") survived the full run whose verdict was closure mesh 800.  The
# tests below pin summary.md to metrics.json and regression-test the ingest
# so a stale summary can never survive an ingest again.

SUMMARY = Path("docs/benchmarks/results/lax_resolution_closure/summary.md")


def test_summary_md_verdict_matches_the_gate(metrics):
    """summary.md is a view of metrics.json: the verdict line must agree.

    The verdict line is rendered by ``benchmarks.lax_resolution_closure.
    render_summary`` as ``Verdict: **<status>**  (closure mesh = <N>)``;
    asserting the exact line (not a substring) is what catches a stale file.
    """
    assert SUMMARY.exists(), (
        f"committed summary {SUMMARY} is missing; the artifact and its one-line "
        "summary are tracked together, so absence is a defect"
    )
    text = SUMMARY.read_text(encoding="utf-8")
    gate = metrics["gate"]
    mesh = "None" if gate["closure_mesh"] is None else str(gate["closure_mesh"])
    expected = f"Verdict: **{gate['status']}**  (closure mesh = {mesh})"
    assert expected in text, (
        f"summary.md does not carry the artifact verdict line {expected!r}; "
        "the file is stale relative to metrics.json"
    )


def test_summary_md_tables_match_metrics(metrics):
    """Every ladder/arm row of summary.md must equal the artifact numbers."""
    assert SUMMARY.exists(), (
        f"committed summary {SUMMARY} is missing; the artifact and its one-line "
        "summary are tracked together, so absence is a defect"
    )
    text = SUMMARY.read_text(encoding="utf-8")
    for e in metrics["ladder"]:
        row = (f"| {e['cells']} | {e['l2']['density']:.4%} | "
               f"{e['l2']['velocity']:.4%} | {e['l2']['pressure']:.4%} | "
               f"{e['l1']['density']:.4%} | {e['band_l1']['density']:.4%} |")
        assert row in text, f"ladder row for N={e['cells']} missing/stale"
    for a in metrics["arms"]:
        row = (f"| {a['limiter']} | {a['l2']['density']:.4%} | "
               f"{a['l2']['velocity']:.4%} | {a['l2']['pressure']:.4%} |")
        assert row in text, f"arm row for limiter={a['limiter']} missing/stale"


def test_ingest_refreshes_a_stale_summary(tmp_path, monkeypatch, metrics):
    """An ingest overwrites a stale summary.md instead of skipping it.

    Reproduces the m22300 defect: a stale summary (closure mesh 200) is
    pre-placed, then a full-run digest is ingested; the refreshed summary
    must carry the ingested closure mesh, not the stale one.
    """
    import scripts.ingest_lax_closure_run as ingest

    out = tmp_path / "out"
    out.mkdir(parents=True)
    monkeypatch.setattr(ingest, "OUT", out)
    monkeypatch.setattr(ingest, "render_doc", lambda: None)
    (out / "summary.md").write_text(
        "# CASE-LAX-DG resolution closure\n"
        "verdict PASS, closure mesh 200\n", encoding="utf-8")

    digest = {k: metrics[k] for k in
              ("ladder", "arms", "gate", "wall_seconds", "generated_at_utc")}
    record = {"job_id": "cjTEST", "node": "nTEST", "rc": 0,
              "output_tail": "DIGEST" + json.dumps(digest)}
    log = tmp_path / "job.log"
    log.write_text(json.dumps(record) + "\n", encoding="utf-8")

    assert ingest.main([str(log)]) == 0
    refreshed = (out / "summary.md").read_text(encoding="utf-8")
    assert "closure mesh 200" not in refreshed
    assert f"closure mesh = {metrics['gate']['closure_mesh']}" in refreshed
    # the ingested metrics.json must be self-contained for render_summary
    reingested = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert reingested["out_dir"] == str(out)


# ------------------------------- m23026 (reopened #676): code / doc / report drift
#
# The tests above pin the *artifact* to the published conclusions.  What they
# cannot see is the other half of the contract: the driver that produced the
# artifact, the human-readable views rendered from it, and the dev report that
# describes this suite.  Each of those can drift away from ``metrics.json``
# while every assertion above stays green:
#
# * editing ``CLOSURE_LADDER`` / ``DEGREE`` / ``ARM_CELLS`` / ``LIMITER_ARMS``
#   in ``benchmarks/lax_resolution_closure.py`` without re-running leaves a
#   committed artifact measured under the *old* configuration;
# * editing ``metrics.json`` by hand (or re-ingesting a run) leaves the case
#   doc and ``summary.md`` quoting the previous verdict;
# * editing this file leaves the dev report quoting tests that no longer exist.
#
# The guards below close those holes without re-running the solver: the views
# are re-rendered from the committed artifact and compared byte-for-byte, the
# driver constants are compared to the artifact's own ``config``, and the dev
# report's table is parsed and compared to the artifact numbers.

CASE_DOC = Path("docs/CASE-LAX-DG-resolution-closure.md")
DEV_REPORT = Path("dev_report_tensordg-lax-resolution-closure-tests.md")
LADDER_TABLE_START = "<!-- LADDER-TABLE-START -->"
LADDER_TABLE_END = "<!-- LADDER-TABLE-END -->"


def test_committed_views_and_figures_are_on_disk(metrics):
    """Every file the closure study publishes must be committed and present."""
    assert CASE_DOC.exists(), f"committed case doc {CASE_DOC} is missing"
    assert DEV_REPORT.exists(), f"committed dev report {DEV_REPORT} is missing"
    assert SUMMARY.exists(), f"committed summary {SUMMARY} is missing"
    assert metrics["figures"], "artifact declares no figures; expected the ladder/arms pair"
    for figure in metrics["figures"]:
        path = METRICS.parent / "figures" / figure["name"]
        assert path.exists() and path.stat().st_size > 0, f"figure {path} missing/empty"


def test_driver_constants_and_artifact_config_agree(metrics):
    """The driver's registered configuration must be the one the artifact measured.

    ``metrics.json`` is only evidence for the configuration recorded in its own
    ``config`` block; if the driver constants were edited afterwards the
    artifact no longer describes the code, and the ladder in this file would be
    pinning a stale study.
    """
    import benchmarks.lax_resolution_closure as driver
    from benchmarks.lax_case_pool import CFL

    cfg = metrics["config"]
    assert tuple(driver.CLOSURE_LADDER) == tuple(cfg["ladder"]), (
        f"driver CLOSURE_LADDER {driver.CLOSURE_LADDER} != artifact ladder {cfg['ladder']}"
    )
    assert driver.DEGREE == cfg["degree"]
    assert driver.ARM_CELLS == cfg["arm_cells"]
    assert tuple("none" if a is None else a for a in driver.LIMITER_ARMS) == tuple(
        cfg["limiter_arms"]
    )
    assert CFL == cfg["cfl"], f"driver CFL {CFL} != artifact CFL {cfg['cfl']}"
    assert tuple(driver.FIELDS) == FIELDS
    # the local pins must not be a third, quietly different configuration
    assert tuple(LADDER) == tuple(cfg["ladder"])
    assert ARMS == tuple(cfg["limiter_arms"])


def test_case_doc_is_a_rendered_view_of_the_artifact(metrics):
    """``docs/CASE-LAX-DG-resolution-closure.md`` must equal its own renderer.

    The doc claims to be a view of ``metrics.json`` ("nothing is transcribed",
    per the driver docstring).  Re-rendering from the committed artifact and
    comparing byte-for-byte is what makes that claim testable: a re-run that
    refreshes the JSON but not the prose fails here instead of shipping two
    disagreeing numbers.
    """
    import scripts.render_lax_closure_doc as renderer

    rendered = renderer.render(metrics)
    committed = CASE_DOC.read_text(encoding="utf-8")
    assert rendered == committed, (
        "docs/CASE-LAX-DG-resolution-closure.md is stale relative to metrics.json; "
        "re-render with scripts.render_lax_closure_doc"
    )


def test_summary_md_is_exactly_the_rendered_view(metrics, tmp_path):
    """``summary.md`` must equal ``render_summary`` output, not merely contain it.

    ``test_summary_md_tables_match_metrics`` checks row substrings; this test
    also catches a stale verdict header, a leftover section from an earlier
    ``--quick`` run, or a row for a mesh that is no longer in the ladder.
    """
    import benchmarks.lax_resolution_closure as driver

    copy = dict(metrics)
    copy["out_dir"] = str(tmp_path)
    rendered = driver.render_summary(copy)
    committed = SUMMARY.read_text(encoding="utf-8")
    assert rendered.rstrip("\n") == committed.rstrip("\n"), (
        "summary.md differs from render_summary(metrics.json); it is stale"
    )


def test_dev_report_test_inventory_matches_the_module():
    """Every test the dev report names must exist, and the count must be right.

    The reopened #676 report drifted from this suite once already (it described
    10 tests whose names no longer existed).  Binding the report to the module
    keeps the delivery document honest without hand-checking it.
    """
    import re

    report = DEV_REPORT.read_text(encoding="utf-8")
    defined = set(re.findall(r"^def (test_[a-z0-9_]+)\(", Path(__file__).read_text(encoding="utf-8"),
                             flags=re.MULTILINE))
    quoted = set(re.findall(r"`(test_[a-z0-9_]+)`", report))
    assert quoted, "dev report names no tests at all"
    unknown = sorted(quoted - defined)
    assert not unknown, f"dev report names tests that do not exist: {unknown}"
    match = re.search(r"\*\*(\d+) 条测试\*\*", report)
    assert match, "dev report must state its test count as '**N 条测试**'"
    assert int(match.group(1)) == len(defined), (
        f"dev report claims {match.group(1)} tests, module defines {len(defined)}"
    )


def test_dev_report_ladder_table_matches_the_artifact(metrics):
    """The dev report's ladder table is parsed and compared to the artifact.

    The table between the two HTML markers is the only place in the report that
    quotes the numbers; rounding to three decimals is allowed, so the tolerance
    is half a unit in the last printed place.
    """
    report = DEV_REPORT.read_text(encoding="utf-8")
    assert LADDER_TABLE_START in report and LADDER_TABLE_END in report, (
        "dev report must delimit its number table with the marker comments"
    )
    body = report.split(LADDER_TABLE_START, 1)[1].split(LADDER_TABLE_END, 1)[0]
    rows = [line for line in body.splitlines() if line.startswith("|")]
    assert len(rows) == len(LADDER) + 2, f"unexpected ladder table shape: {rows}"
    seen = []
    for line in rows[2:]:  # skip header + separator
        cells = [c.strip() for c in line.strip("|").split("|")]
        assert len(cells) == 4, f"malformed row {line!r}"
        n = int(cells[0])
        seen.append(n)
        leg = _find_leg(metrics, n)
        for field, printed in zip(FIELDS, cells[1:]):
            value = 100.0 * leg["l2"][field]
            assert abs(float(printed.rstrip("%").strip()) - value) <= 5e-4, (
                f"report row N={n} {field}: printed {printed} %, artifact {value:.6f} %"
            )
    assert seen == list(LADDER)
