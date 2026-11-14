"""Acceptance / invariant tests for CASE-BIAXIAL-SHEAR-BAND.

The case itself is an honest FAIL on the band-inclination acceptance (the
elastoplastic specimen softens almost uniformly at the tested resolution, so no
localized band is measured); what *is* established here is the verification of
the two components that the case was built on:

* the Mohr-Coulomb return mapping (yield-surface residual, apex cap, flow rule),
* the Rice/Rudnicki-Rice shear-band bifurcation analysis, which reproduces the
  literature law ``theta = 45 + (phi+psi)/4`` to well under 1 % and degenerates
  to the Mohr-Coulomb slip plane ``45 + phi/2`` for associated flow.

The gates below are the ones that pass; nothing asserts the failed acceptance.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensordg.plasticity_2d import (PlaneStrainMC, band_angle_closed_form,
                                    localization_angle, mean_deviatoric,
                                    return_map, biaxial_elastic_exact)
from tensordg.biaxial_shear_band import band_angle_from_field, structure_tensor_angle

ROOT = Path(__file__).resolve().parent.parent
STATUS = ROOT / "validation" / "biaxial_shear_band" / "status.json"
DOC = ROOT / "docs" / "CASE-BIAXIAL-SHEAR-BAND.md"


def _mat(phi_deg=40.0, psi_deg=15.0, **kw):
    return PlaneStrainMC(E=30.0e6, nu=0.3, rho=2000.0,
                         phi=math.radians(phi_deg), psi=math.radians(psi_deg),
                         **kw)


class TestLocalizationAnalysis:
    """Rice (1976) bifurcation angle vs the literature closed form."""

    @pytest.mark.parametrize("phi,psi", [(40.0, 15.0), (30.0, 0.0), (35.0, 10.0)])
    def test_matches_literature_law(self, phi, psi):
        mat = _mat(phi, psi)
        theta, roots = localization_angle(mat)
        ref = band_angle_closed_form(mat.phi, mat.psi)       # 45 + (phi+psi)/4
        assert abs(theta - ref)/ref < 0.01, f"{theta:.3f} vs {ref:.3f}"
        assert len(roots) >= 1                                # conjugate band pair

    def test_associative_flow_recovers_coulomb_slip_plane(self):
        mat = _mat(40.0, 40.0)                                # psi == phi
        theta, _ = localization_angle(mat)
        ref = 45.0 + 40.0/2.0                                 # 65 deg
        assert abs(theta - ref)/ref < 1.0e-3, f"{theta:.4f} vs {ref}"

    def test_closed_form_limits(self):
        assert band_angle_closed_form(math.radians(30.0), 0.0) == pytest.approx(52.5)
        assert band_angle_closed_form(math.radians(30.0), math.radians(30.0)) == \
            pytest.approx(60.0)


class TestReturnMapping:
    """Invariants of the plane-strain Mohr-Coulomb return mapping."""

    def test_yield_surface_residual(self):
        mat = _mat()
        rng = np.random.default_rng(7)
        n = 3000
        sxx = rng.uniform(-4e5, 4e5, n)
        syy = rng.uniform(-4e5, 4e5, n)
        sxy = rng.uniform(-4e5, 4e5, n)
        c = np.full(n, 350.0e3)
        m, q = mean_deviatoric(sxx, syy, sxy)
        F0 = q + m*mat.sin_phi - c*mat.cos_phi
        sx, sy, sxyo, dgp = return_map(mat, sxx, syy, sxy, c)
        assert np.isfinite(dgp).all()
        m1, q1 = mean_deviatoric(sx, sy, sxyo)
        F1 = q1 + m1*mat.sin_phi - c*mat.cos_phi
        act = F0 > 0
        # plastic points land on (or inside, for the apex branch) the surface
        assert np.median(F1[act]) < 1e-6*c.max()
        assert (F1[act] > 1e-2*c.max()).mean() < 0.02

    def test_elastic_when_inside(self):
        mat = _mat()
        sxx = np.array([-5.0e4]); syy = np.array([-5.0e4]); sxy = np.array([0.0])
        c = np.array([350.0e3])
        sx, sy, sxyo, dgp = return_map(mat, sxx, syy, sxy, c)
        assert dgp[0] == 0.0
        assert sx[0] == sxx[0] and sy[0] == syy[0]

    def test_apex_is_capped(self):
        mat = _mat()
        # deep in the tension region -> must be projected onto m = c cot(phi)
        c = np.array([350.0e3])
        m_tip = c[0]*mat.cos_phi/mat.sin_phi
        sxx = np.array([1.2*m_tip]); syy = np.array([1.2*m_tip]); sxy = np.array([1.0e4])
        sx, sy, sxyo, dgp = return_map(mat, sxx, syy, sxy, c)
        assert abs(0.5*(sx[0]+sy[0]) - m_tip)/m_tip < 1e-12
        assert abs(sxyo[0]) == 0.0
        assert dgp[0] > 0.0 and np.isfinite(dgp[0])

    def test_uniaxial_compression_path_is_monotone(self):
        mat = _mat()
        syy = np.full(8, -5.0e4)
        sxx = -5.0e4 - np.array([0.0, 5e4, 1e5, 2e5, 3e5, 4e5, 5e5, 6e5])
        sxy = np.zeros(8)
        c = np.full(8, 350.0e3)
        _, _, _, dgp = return_map(mat, sxx, syy, sxy, c)
        assert np.all(np.diff(dgp) >= -1e-12)

    def test_softening_law(self):
        mat = _mat(gp_ref=0.02)
        gp = np.array([0.0, 0.01, 0.02, 0.05])
        c = mat.cohesion(gp)
        assert c[0] == pytest.approx(mat.c0)
        assert c[2] == pytest.approx(mat.c_res)
        assert c[3] == pytest.approx(mat.c_res)
        assert np.all(np.diff(c) <= 0.0)


class TestBandMeasurement:
    """The band-orientation estimator must recover a synthetic band."""

    @pytest.mark.parametrize("angle_deg", [50.0, 58.75, 65.0, 70.0])
    def test_recovers_synthetic_band(self, angle_deg):
        n = 80
        x = np.linspace(0.0, 1.0, n)
        y = np.linspace(0.0, 2.0, n)
        X, Y = np.meshgrid(x, y, indexing="ij")
        t = math.tan(math.radians(angle_deg))
        # a thin band through the centre with that inclination, plus noise
        dist = np.abs((X-0.5) - (Y-1.0)/t)/math.sqrt(1.0+1.0/t**2)
        field = np.exp(-(dist/0.05)**2)
        ang, npts, contrast = band_angle_from_field(field, x, y)
        assert npts > 20 and contrast > 1.5
        assert abs(ang-angle_deg) < 2.0, f"{ang:.2f} vs {angle_deg}"
        ang_st = structure_tensor_angle(field, x[1]-x[0], y[1]-y[0])
        assert abs(ang_st-angle_deg) < 5.0

    def test_no_structure_returns_nan(self):
        field = np.ones((20, 20))
        ang, npts, contrast = band_angle_from_field(field, np.arange(20)*0.1,
                                                    np.arange(20)*0.1)
        assert math.isnan(ang)


class TestElasticReference:
    def test_exact_solution_is_consistent(self):
        mat = _mat()
        sol = biaxial_elastic_exact(-1e-3, 5.0e4, mat=mat)
        assert sol["sig_xx"] == pytest.approx(-5.0e4, rel=1e-12)
        # plane-strain volumetric relation: m = a (eps_lateral + eps_axial)/2
        assert 0.5*(sol["sig_xx"] + sol["sig_yy"]) == pytest.approx(
            mat.bulk2d*0.5*(sol["eps_lateral"] + sol["eps_axial"]), rel=1e-10)


class TestCaseArtifacts:
    """The case pool artifact exists, is self-consistent and is honestly FAIL."""

    @pytest.mark.skipif(not STATUS.exists(), reason="status.json not generated")
    def test_status_artifact(self):
        st = json.loads(STATUS.read_text())
        assert st["case"] == "CASE-BIAXIAL-SHEAR-BAND"
        assert st["localization"]["rel_error_vs_closed_form"] < 0.01
        assert st["localization"]["associative_rel_error"] < 1e-3
        b = st["biaxial"]
        assert b["ke_over_ie_median"] < 0.05, "run is not quasi-static"
        # honest bookkeeping: the acceptance is recorded as it came out
        assert isinstance(b["acceptance_pass"], bool)
        if not b["localized"]:
            assert math.isnan(b["band_angle_measured_deg"])

    @pytest.mark.skipif(not DOC.exists(), reason="case document not generated")
    def test_document_is_programmatically_injected(self):
        txt = DOC.read_text()
        st = json.loads(STATUS.read_text())
        assert "① 案例介绍" in txt and "⑥ 误差判定" in txt
        theta = st["localization"]["theta_closed_form_deg"]
        assert f"{theta:.2f}" in txt