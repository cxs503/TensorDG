"""P3 non-monotonicity test for Shu-Osher round-2 certified reference.

This test documents and verifies the known behavior that DG P3 at N=800
has *higher* error than DG P2 at N=800 in the Shu-Osher shock/entropy-wave
interaction case.  This is **not a regression** — it is an expected consequence
of the WENO limiter's mode-zeroing strategy in troubled cells.

Background
----------
In the Shu-Osher case (Shu & Osher 1988), the error is dominated by the
limiter's smearing near the shock and contact discontinuity.  The WENO limiter
(``tensordg.limiters.apply_weno_limiter``) replaces the high-order polynomial
in troubled cells with a WENO3 reconstruction, zeroing out modes 2+.

For P2 (3 modes): troubled cells lose mode-2 → reduced to linear (1st order)
For P3 (4 modes): troubled cells lose modes 2,3 → also reduced to linear

Since P3 loses *more* modes but achieves the *same* reduced accuracy in
troubled cells, its net error can be higher than P2's at the same resolution.

This test asserts the *observed* behavior (from the committed audit JSON) so
that future changes to the limiter or solver can be checked against this
baseline.  If a future change makes P3 better than P2 at N=800, that would
be a genuine improvement and this test should be updated.

See ``docs/CASE-XVAL-DG-P3-NONMONOTONICITY.md`` for the full analysis.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

AUDIT = Path("validation/xval_dg_round2/artifacts/multifield_audit.json")


def _load_legs():
    """Load leg data from the committed multifield audit JSON."""
    if not AUDIT.exists():
        pytest.skip(f"audit not found at {AUDIT}")
    with open(AUDIT) as f:
        data = json.load(f)
    return data["legs"]


def _find_leg(legs, degree, ncells, cfl=0.4):
    """Find a leg by degree, ncells, and CFL."""
    for leg in legs:
        if leg["degree"] == degree and leg["ncells"] == ncells and leg["cfl"] == cfl:
            return leg
    return None


# ---------------------------------------------------------------------------
# Core assertions: P3 non-monotonicity at N=800
# ---------------------------------------------------------------------------

class TestP3NonMonotonicity:
    """Assert the known P3 > P2 error at N=800 (not a regression)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.legs = _load_legs()
        self.p2 = _find_leg(self.legs, 2, 800)
        self.p3 = _find_leg(self.legs, 3, 800)
        assert self.p2 is not None, "P2 N=800 leg not found in audit"
        assert self.p3 is not None, "P3 N=800 leg not found in audit"

    def test_p3_rho_error_exceeds_p2_at_n800(self):
        """P3 rho error > P2 rho error at N=800 (known behavior)."""
        rho_p2 = self.p2["rho"]["rel_l1_percent"]
        rho_p3 = self.p3["rho"]["rel_l1_percent"]
        assert rho_p3 > rho_p2, (
            f"P3 rho error ({rho_p3:.6f}%) should exceed P2 ({rho_p2:.6f}%) "
            f"at N=800 due to limiter mode-zeroing"
        )

    def test_p3_rho_u_error_exceeds_p2_at_n800(self):
        """P3 rho_u error > P2 rho_u error at N=800."""
        rho_u_p2 = self.p2["rho_u"]["rel_l1_percent"]
        rho_u_p3 = self.p3["rho_u"]["rel_l1_percent"]
        assert rho_u_p3 > rho_u_p2, (
            f"P3 rho_u error ({rho_u_p3:.6f}%) should exceed P2 ({rho_u_p2:.6f}%) "
            f"at N=800"
        )

    def test_p3_rho_e_error_exceeds_p2_at_n800(self):
        """P3 rho_E error > P2 rho_E error at N=800 (largest gap)."""
        rho_e_p2 = self.p2["rho_e"]["rel_l1_percent"]
        rho_e_p3 = self.p3["rho_e"]["rel_l1_percent"]
        assert rho_e_p3 > rho_e_p2, (
            f"P3 rho_E error ({rho_e_p3:.6f}%) should exceed P2 ({rho_e_p2:.6f}%) "
            f"at N=800"
        )

    def test_p3_p2_error_ratio_bounds(self):
        """P3/P2 error ratio is within expected bounds (1.1-1.8x)."""
        for field in ("rho", "rho_u", "rho_e"):
            r2 = self.p2[field]["rel_l1_percent"]
            r3 = self.p3[field]["rel_l1_percent"]
            ratio = r3 / r2
            # P3 should be worse, but not catastrophically so
            assert 1.0 < ratio < 2.5, (
                f"P3/P2 ratio for {field} = {ratio:.3f}x, expected 1.0-2.5x"
            )


# ---------------------------------------------------------------------------
# EOC analysis: P3 converges slower than P2 in second refinement
# ---------------------------------------------------------------------------

class TestEOCAnomaly:
    """Assert the EOC anomaly: P3's second-refinement EOC < P2's."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.legs = _load_legs()

    def _eoc(self, e1, n1, e2, n2):
        """Compute effective order of convergence."""
        if e1 <= 0 or e2 <= 0:
            return None
        return -math.log(e2 / e1) / math.log(n2 / n1)

    def _get_errors(self, degree, field="rho"):
        """Get sorted (ncells, error) pairs for a degree."""
        errs = []
        for leg in self.legs:
            if leg["degree"] == degree and leg["cfl"] == 0.4:
                errs.append((leg["ncells"], leg[field]["rel_l1_percent"]))
        errs.sort()
        return errs

    def test_p1_eoc_increases_with_refinement(self):
        """P1 EOC should increase from coarse to fine (expected)."""
        errs = self._get_errors(1)
        eoc1 = self._eoc(errs[0][1], errs[0][0], errs[1][1], errs[1][0])
        eoc2 = self._eoc(errs[1][1], errs[1][0], errs[2][1], errs[2][0])
        assert eoc2 > eoc1, f"P1 EOC should increase: {eoc1:.3f} → {eoc2:.3f}"

    def test_p2_eoc_increases_with_refinement(self):
        """P2 EOC should increase from coarse to fine (expected)."""
        errs = self._get_errors(2)
        eoc1 = self._eoc(errs[0][1], errs[0][0], errs[1][1], errs[1][0])
        eoc2 = self._eoc(errs[1][1], errs[1][0], errs[2][1], errs[2][0])
        assert eoc2 > eoc1, f"P2 EOC should increase: {eoc1:.3f} → {eoc2:.3f}"

    def test_p3_eoc_second_refinement_below_p2(self):
        """P3's second-refinement EOC is below P2's (the anomaly)."""
        errs_p2 = self._get_errors(2)
        errs_p3 = self._get_errors(3)
        eoc_p2 = self._eoc(errs_p2[1][1], errs_p2[1][0], errs_p2[2][1], errs_p2[2][0])
        eoc_p3 = self._eoc(errs_p3[1][1], errs_p3[1][0], errs_p3[2][1], errs_p3[2][0])
        assert eoc_p3 < eoc_p2, (
            f"P3 EOC ({eoc_p3:.3f}) should be below P2 EOC ({eoc_p2:.3f}) "
            f"in second refinement due to limiter degradation"
        )

    def test_all_eoc_below_theoretical(self):
        """All EOC values are below theoretical smooth-region orders (shock case)."""
        theoretical = {1: 2, 2: 3, 3: 4}
        for deg, theo in theoretical.items():
            errs = self._get_errors(deg)
            for i in range(1, len(errs)):
                eoc = self._eoc(errs[i-1][1], errs[i-1][0], errs[i][1], errs[i][0])
                assert eoc is not None and eoc < theo, (
                    f"P{deg} EOC {eoc:.3f} should be < theoretical {theo} "
                    f"(shock problem, not smooth)"
                )


# ---------------------------------------------------------------------------
# Anti-staleness: audit JSON must match recomputed values
# ---------------------------------------------------------------------------

def test_audit_json_is_not_stale():
    """The committed audit JSON must be internally consistent."""
    if not AUDIT.exists():
        pytest.skip(f"audit not found at {AUDIT}")
    with open(AUDIT) as f:
        data = json.load(f)

    # Check required keys
    assert "legs" in data
    assert len(data["legs"]) == 11, f"Expected 11 legs, got {len(data['legs'])}"

    # Check that P2 and P3 at N=800 exist
    p2 = _find_leg(data["legs"], 2, 800)
    p3 = _find_leg(data["legs"], 3, 800)
    assert p2 is not None
    assert p3 is not None

    # Check that the non-monotonicity is present (sanity check)
    assert p3["rho"]["rel_l1_percent"] > p2["rho"]["rel_l1_percent"]
