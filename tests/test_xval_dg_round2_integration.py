"""Integration gates for XVAL-DG-ROUND2-CERTIFIED-REFERENCE (mission deliverable).

What this file pins, and why these assertions are not "the code ran"
--------------------------------------------------------------------
The round-2 audit (``docs/AUDIT-m17539-XVAL-DG-ROUND2.md``) accepted the *unit*
level (``tests/test_certified_reference.py``: the engine's own invariants) but
blocked integration: the engine could not be reached from the package root, no
benchmark driver composed engine + case + oracle, and nothing tied the numbers
to the platform contract.  This module closes that with end-to-end gates:

``T1`` the benchmark driver runs from the CLI, writes its artifacts, and exits
      0 only when all four predeclared decisions pass (``--require-accepted``);
``T2`` every DG leg is under the strict 3 % standard and the reference is
      usable (band / gate <= 1/3);
``T3`` the certified reference agrees with **closed-form** sources it did not
      come from (Hugoniot plateau, Hugoniot shock position, exact mass budget);
``T4`` no DG family may certify itself (all three must be refused), so T2/T3
      cannot be self-referential;
``T5`` the relative L1 error of a leg is recomputable with an independent
      coarsen/rel-L1 implementation (no engine helper) -- the number is not a
      property of the engine's code path;
``T6`` the platform documents pass the *platform's own* TensorBench contract
      (spec keys, result schema, pointer resolution, metric evaluation) and the
      driver refuses to post without a cluster job id;
``T7`` the artifacts are deterministic (two runs, identical bytes).

All of it runs on the committed raw profiles
(``validation/repro_shu_osher/artifacts/raw``) in seconds -- no solver rerun.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "validation" / "repro_shu_osher" / "artifacts" / "raw"
POST = ROOT / "validation" / "xval_dg_round2" / "post_platform.py"

sys.path.insert(0, str(ROOT))

from benchmarks import xval_dg_round2_certified as xval  # noqa: E402

pytestmark = pytest.mark.skipif(
    not RAW.exists(), reason="committed round-2 raw profiles are required")


@pytest.fixture(scope="module")
def audit() -> dict:
    return xval.cross_validate(RAW)


@pytest.fixture(scope="module")
def post_module():
    spec = importlib.util.spec_from_file_location("xval_post_platform", POST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ T1 CLI
def test_cli_writes_artifacts_and_passes_all_gates(tmp_path):
    rc = xval.main(["--rawdir", str(RAW), "--outdir", str(tmp_path), "--require-accepted"])
    assert rc == 0
    js, md = tmp_path / "xval_round2_audit.json", tmp_path / "xval_round2_audit.md"
    assert js.exists() and md.exists()
    payload = json.loads(js.read_text())
    assert payload["accepted"] is True
    assert {k for k in payload if k.startswith("D")} == {
        "D1_reference_usable", "D2_legs_under_strict_gate",
        "D3_reference_vs_oracle", "D4_no_self_certification"}
    assert "**accepted = True**" in md.read_text()


def test_cli_json_stdout_is_the_platform_document(tmp_path):
    block = xval.platform_block(xval.cross_validate(RAW))
    doc = json.loads(json.dumps(block["stdout_document"]))
    got = xval.main(["--rawdir", str(RAW), "--json-stdout", "--no-write"])
    assert got == 0
    # the emitted document must contain every pointer the spec uses
    for metric in block["metrics"]:
        for pointer in (metric["got_pointer"], metric["ref_pointer"]):
            node = doc
            for part in pointer.split("."):
                node = node[part]
            assert isinstance(node, float)


def test_a_failing_gate_makes_the_cli_exit_nonzero(tmp_path):
    """An impossible oracle tolerance must turn the run red, not silently pass."""
    rc = xval.main(["--rawdir", str(RAW), "--outdir", str(tmp_path),
                    "--anchor-tolerance", "1e-12", "--require-accepted"])
    assert rc == 1
    assert json.loads((tmp_path / "xval_round2_audit.json").read_text())["accepted"] is False


# ------------------------------------------------------- T2/T3/T4 decisions
def test_reference_is_usable_and_every_leg_meets_the_strict_standard(audit):
    d1, d2 = audit["D1_reference_usable"], audit["D2_legs_under_strict_gate"]
    assert d1["verdict"] == "CERTIFIED"
    assert d1["band_share_of_gate"] <= audit["thresholds"]["u_share_max"]
    assert d2["n_legs"] == 11 and len(audit["legs"]) == 11
    for leg in audit["legs"]:
        assert leg["rel_l1_percent"] < audit["thresholds"]["strict_percent"], leg["configuration"]
        assert leg["decision_5pct"] == "PASS", leg["configuration"]
        assert leg["decision_3pct"] == "PASS", leg["configuration"]
    assert d2["worst_rel_l1_percent"] < 3.0
    assert d2["pass"] and audit["accepted"]


def test_reference_agrees_with_closed_form_oracles(audit):
    rows = {row["name"]: row for row in audit["D3_reference_vs_oracle"]["rows"]}
    # plateau density is the Mach-3 Hugoniot rho_L, shock position the Hugoniot
    # x_L + W t, mass the exact inflow budget: none of them comes from the engine
    assert rows["plateau_rho"]["rel_error_percent"] < 1.0
    assert rows["shock_position"]["rel_error_percent"] < 1.0
    assert rows["mass_integral"]["rel_error_percent"] < 1.0
    from tensordg.cases.shock_entropy import LEFT_STATE, rankine_hugoniot_anchor
    # the oracle plateau is the Riemann datum rho_L = 3.857143, and that datum is
    # itself the Mach-3 Hugoniot state (checked to 1e-6, not assumed)
    assert rows["plateau_rho"]["ref"] == pytest.approx(LEFT_STATE[0], rel=1e-15)
    hug = rankine_hugoniot_anchor()["hugoniot_left_state_from_mach3"]
    assert LEFT_STATE[0] == pytest.approx(hug["rho"], rel=1e-6)
    assert rows["shock_position"]["ref"] == pytest.approx(
        rankine_hugoniot_anchor()["shock_position_analytic"], rel=1e-15)
    assert audit["oracle"]["source"].startswith("tensordg.cases.shock_entropy")


def test_no_dg_family_may_certify_itself(audit):
    d4 = audit["D4_no_self_certification"]
    assert d4["pass"] is True
    assert set(d4["ladders"].values()) == {"WITHHELD_ORDER_TOO_LOW"}


# ------------------------------------------------- T5 independent recompute
def _my_coarsen(values: np.ndarray, factor: int) -> np.ndarray:
    return values.reshape(values.size // factor, factor).mean(axis=1)


def test_leg_error_is_reproducible_without_the_engine(audit):
    """Recompute one leg's rel L1 from the raw npz with local helpers only."""
    leg = next(lg for lg in audit["legs"] if lg["configuration"] == "DG P1 N=200 CFL=0.4")
    ref = xval.certified_reference(RAW)["rho"]
    with np.load(RAW / leg["file"], allow_pickle=True) as d:
        cand = np.asarray(d["rho"], dtype=float)
    factor = ref.size // cand.size
    assert factor * cand.size == ref.size
    ref_c = _my_coarsen(np.asarray(ref, dtype=float), factor)
    mine = 100.0 * np.sum(np.abs(cand - ref_c)) / np.sum(np.abs(ref_c))
    assert mine == pytest.approx(leg["rel_l1_percent"], rel=1e-12)
    assert mine < 3.0


def test_metrics_are_measured_on_the_reference_grid(audit):
    ref = xval.certified_reference(RAW)
    assert ref["rho"].size == audit["D1_reference_usable"]["reference_ncells"]
    x = ref["x"]
    assert x.size == ref["rho"].size
    assert x[0] > -5.0 and x[-1] < 5.0
    # a certified profile is a convex combination of cell averages: no overshoot
    assert ref["rho"].min() > 0.0
    assert ref["rho"].max() < audit["oracle"]["plateau_rho"] + 1.0


# ----------------------------------------------------- T6 platform contract
def test_platform_documents_pass_the_platform_contract(post_module):
    docs = post_module.build(18730)
    report = post_module.validate(docs)
    assert report["mode"] == "platform_contract", (
        "the platform TensorBench contract module must be importable for this gate")
    assert report["all_passed"], report["rows"]
    assert report["result_meta"]["benchmark_id"] == xval.BENCH_ID
    assert {row["metric"] for row in report["rows"]} == {
        "plateau_rho", "shock_position", "mass_integral"}
    allowed = {"name", "got_pointer", "ref_pointer", "tolerance_rel", "unit", "better"}
    for metric in docs["spec"]["evaluation"]["metrics"]:
        assert set(metric) <= allowed
        assert metric["tolerance_rel"] == pytest.approx(
            xval.ANCHOR_TOL_PERCENT / 100.0)


def test_post_platform_refuses_to_post_without_a_cluster_job(post_module, tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        post_module.main(["--mission", "18730", "--post",
                          "--out", str(tmp_path / "payload.json")])
    assert "--job-id" in str(excinfo.value)
    assert (tmp_path / "payload.json").exists()


# -------------------------------------------------------------- T7 determinism
def test_artifacts_are_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert xval.main(["--rawdir", str(RAW), "--outdir", str(a)]) == 0
    assert xval.main(["--rawdir", str(RAW), "--outdir", str(b)]) == 0
    for name in ("xval_round2_audit.json", "xval_round2_audit.md"):
        assert (a / name).read_bytes() == (b / name).read_bytes()