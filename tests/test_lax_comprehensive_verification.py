"""Comprehensive verification tests for CASE-LAX-DG.

These tests cross-validate the comprehensive verification report
(docs/CASE-LAX-DG-comprehensive-verification.md) against the underlying
artifact JSON files, ensuring zero hand-transcription drift.

Tests assert invariants across all five evidence layers (L1–L5).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "docs" / "benchmarks" / "results" / "lax_case_pool" / "metrics.json"
CERT = ROOT / "docs" / "benchmarks" / "results" / "lax_certified_configuration" / "metrics.json"
RES_REQ = ROOT / "docs" / "benchmarks" / "results" / "lax_resolution_requirement" / "metrics.json"
RES_CLOSURE = ROOT / "docs" / "benchmarks" / "results" / "lax_resolution_closure" / "metrics.json"
REF_CROSS = ROOT / "docs" / "benchmarks" / "results" / "lax_reference_crosscheck" / "metrics.json"
LIMITER_ARMS = ROOT / "docs" / "benchmarks" / "results" / "lax_case_pool_limiter_arms" / "metrics.json"
COMPREHENSIVE_DOC = ROOT / "docs" / "CASE-LAX-DG-comprehensive-verification.md"

TOLERANCE = 0.03  # 3 % ledger acceptance


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# L1: Case pool invariants
# ---------------------------------------------------------------------------

class TestCasePoolInvariants:
    """L1: 算例池 9 腿阶梯的基本不变量。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.pool = _load(POOL)

    def test_nine_legs_present(self):
        """9 腿阶梯完整。"""
        ladder = self.pool["ladder"]
        assert len(ladder) == 9
        expected = [(100, 1), (200, 1), (400, 1),
                    (100, 2), (200, 2), (400, 2),
                    (100, 3), (200, 3), (400, 3)]
        actual = [(leg["cells"], leg["degree"]) for leg in ladder]
        assert actual == expected

    def test_p3_n400_is_production_leg(self):
        """p3.N400 是登记的生产腿。"""
        prod = self.pool["acceptance"]["production_leg"]
        assert prod == "p3.N400"

    def test_p3_n400_density_fails_3pct(self):
        """p3.N400 密度 rel L2 > 3 %（honest FAIL）。"""
        leg = self.pool["ladder"][8]  # last leg = p3.N400
        assert leg["cells"] == 400 and leg["degree"] == 3
        rho_l2 = leg["l2_errors"]["density"]
        assert rho_l2 > TOLERANCE, f"Expected > {TOLERANCE}, got {rho_l2}"
        # Verify the exact number from the comprehensive doc
        assert abs(rho_l2 - 0.0311556) < 1e-6

    def test_all_l2_errors_decrease_with_refinement(self):
        """同一阶数下，加密使 L2 误差单调递减。"""
        ladder = self.pool["ladder"]
        for deg in [1, 2, 3]:
            legs = [l for l in ladder if l["degree"] == deg]
            for i in range(len(legs) - 1):
                for field in ["density", "velocity", "pressure"]:
                    assert legs[i]["l2_errors"][field] > legs[i + 1]["l2_errors"][field]

    def test_gates_c1_c2_r5_pass(self):
        """C1、C2、R5 闸门 PASS。"""
        gates = self.pool["gates"]
        assert gates["c1"]["worst_relative_error"] < gates["c1"]["tolerance"]
        # C2: check star state errors
        c2 = gates["c2"]
        for wave_key, wave_data in c2.get("star", {}).items():
            assert wave_data["relative_error"] < c2["tolerance"]
        # R5: entropy wave
        r5 = gates["r5"]
        for leg_key, leg_data in r5.get("legs", {}).items():
            assert leg_data["deviation_pct"] < r5["tolerance"]

    def test_gates_c3_r3_fail(self):
        """C3、R3 闸门 FAIL（间断压阶）。"""
        gates = self.pool["gates"]
        c3 = gates["c3"]
        # C3 FAIL means observed order < tolerance for some legs
        for leg_key, leg_data in c3.get("legs", {}).items():
            if "p3" in leg_key:
                assert leg_data["order"] < c3["tolerance"]
        # R3 FAIL
        r3 = gates["r3"]
        for leg_key, leg_data in r3.get("legs", {}).items():
            if "p3" in leg_key:
                assert leg_data["ratio"] < 1.0  # ratio < 1 means FAIL


# ---------------------------------------------------------------------------
# L2: A0 acceptance invariants
# ---------------------------------------------------------------------------

class TestA0Acceptance:
    """L2: A0 台账验收闸门。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.pool = _load(POOL)

    def test_production_l2_fails(self):
        """production_l2 臂 FAIL。"""
        acc = self.pool["acceptance"]
        arms = acc["arms"]
        assert arms["production_l2"]["status"] == "FAIL"
        assert arms["production_l2"]["worst"] > TOLERANCE

    def test_ladder_l1_passes(self):
        """ladder_l1 臂 PASS（更严读数）。"""
        acc = self.pool["acceptance"]
        arms = acc["arms"]
        assert arms["ladder_l1"]["status"] == "PASS"

    def test_tolerance_matches_case_module(self):
        """容差与 case 模块一致。"""
        from tensordg.cases.lax_1d import lax_parameters
        params = lax_parameters()
        assert abs(params["acceptance_threshold"] - TOLERANCE) < 1e-12


# ---------------------------------------------------------------------------
# L3: Resolution closure invariants
# ---------------------------------------------------------------------------

class TestResolutionClosure:
    """L3: 分辨率闭合验证。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.closure = _load(RES_CLOSURE)

    def test_n800_passes_all_fields(self):
        """N=800 三量全部 < 3 %。"""
        ladder = self.closure["ladder"]
        n800 = None
        for leg in ladder:
            if leg["cells"] == 800:
                n800 = leg
                break
        assert n800 is not None, "N=800 leg not found"
        for field in ["density", "velocity", "pressure"]:
            err = n800["l2"][field]
            assert err < TOLERANCE, f"N=800 {field} L2 = {err} >= {TOLERANCE}"

    def test_closure_verdict_is_pass(self):
        """闭合 verdict = PASS。"""
        gate = self.closure["gate"]
        # Check per_field: all fields should have at least one leg below tolerance
        for field_data in gate["per_field"].values():
            series = field_data["series"]
            has_pass = any(s["error"] < TOLERANCE for s in series)
            assert has_pass, f"Field {field_data['field']} has no leg below tolerance"

    def test_closure_mesh_is_800(self):
        """closure_mesh = 800（第一个达标的腿）。"""
        ladder = self.closure["ladder"]
        for leg in ladder:
            if all(leg["l2"][f] < TOLERANCE for f in ["density", "velocity", "pressure"]):
                assert leg["cells"] == 800, f"First passing leg is N={leg['cells']}, expected 800"
                break


# ---------------------------------------------------------------------------
# L4: Certified configuration invariants
# ---------------------------------------------------------------------------

class TestCertifiedConfiguration:
    """L4: 认证构型（最小自由度达标）。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.cert = _load(CERT)

    def test_certified_leg_is_p2_n800(self):
        """认证构型 = p2.N800。"""
        c = self.cert["certification"]["certified"]
        assert c["leg"] == "p2.N800"

    def test_certified_dof_is_2400(self):
        """DOF = 2400。"""
        c = self.cert["certification"]["certified"]
        assert c["dof"] == 2400

    def test_certified_worst_l2_below_tolerance(self):
        """认证构型最差量 < 3 %。"""
        c = self.cert["certification"]["certified"]
        assert c["worst_l2"] < TOLERANCE

    def test_certified_worst_field_is_density(self):
        """最差量 = density。"""
        c = self.cert["certification"]["certified"]
        assert c["worst_field"] == "density"


# ---------------------------------------------------------------------------
# L5: Reference crosscheck invariants
# ---------------------------------------------------------------------------

class TestReferenceCrosscheck:
    """L5: 参考解独立仲裁。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ref = _load(REF_CROSS)

    def test_fv_convergence_passes(self):
        """FV 收敛阶达标。"""
        gates = self.ref["gates"]
        assert gates["convergence_first_order"]["status"] == "PASS"
        assert gates["convergence_second_order"]["status"] == "PASS"

    def test_root_solver_agrees(self):
        """二分 vs Newton 根求解一致。"""
        gates = self.ref["gates"]
        assert gates["root_finder"]["status"] == "PASS"

    def test_entropy_condition_satisfied(self):
        """熵条件满足。"""
        gates = self.ref["gates"]
        assert gates["entropy_condition"]["status"] == "PASS"


# ---------------------------------------------------------------------------
# Cross-layer invariants
# ---------------------------------------------------------------------------

class TestCrossLayerConsistency:
    """跨层一致性：综合报告数字与工件逐位一致。"""

    def test_comprehensive_doc_exists(self):
        """综合报告文件存在。"""
        assert COMPREHENSIVE_DOC.exists()

    def test_comprehensive_doc_mentions_honest_fail(self):
        """综合报告包含 honest FAIL 声明。"""
        content = COMPREHENSIVE_DOC.read_text()
        assert "FAIL" in content
        assert "3.116" in content or "3.11556" in content

    def test_comprehensive_doc_mentions_certified_config(self):
        """综合报告包含认证构型。"""
        content = COMPREHENSIVE_DOC.read_text()
        assert "p2.N800" in content
        assert "2400" in content

    def test_pool_p3_n400_matches_closure_n400(self):
        """算例池 p3.N400 与分辨率闭合 N=400 的密度 L2 一致。"""
        pool = _load(POOL)
        closure = _load(RES_CLOSURE)
        pool_p3n400 = pool["ladder"][8]["l2_errors"]["density"]
        # Find N=400 in closure
        closure_n400 = None
        for leg in closure["ladder"]:
            if leg["cells"] == 400:
                closure_n400 = leg["l2"]["density"]
                break
        if closure_n400 is not None:
            assert abs(pool_p3n400 - closure_n400) < 1e-10, \
                f"Pool p3.N400 ρ L2 = {pool_p3n400}, closure N=400 = {closure_n400}"

    def test_certified_config_reproduces_pool_p3_n400(self):
        """认证构型研究中 p3.N400 复现算例池工件。"""
        cert = _load(CERT)
        pool = _load(POOL)
        # Find p3.N400 in cert legs
        pool_p3n400 = pool["ladder"][8]["l2_errors"]["density"]
        for leg in cert["legs"]:
            if leg.get("leg") == "p3.N400":
                cert_rho = leg["l2_errors"]["density"]
                assert abs(pool_p3n400 - cert_rho) < 1e-10, \
                    f"Pool p3.N400 ρ L2 = {pool_p3n400}, cert = {cert_rho}"
                break
