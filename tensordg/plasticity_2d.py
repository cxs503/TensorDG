"""Plane-strain elastoplasticity for cohesive-frictional solids (small strain).

This module supplies the two ingredients that the CASE-BIAXIAL-SHEAR-BAND case
needs, and nothing else:

* :func:`return_map` -- an implicit (radial/secant) return mapping of a
  plane-strain Mohr-Coulomb material with **non-associated** plastic flow
  (dilation angle ``psi``), linear softening of the cohesion and an
  apex/tension cap at ``m = c cot(phi)``.  Stresses are *tension positive*
  (:math:`m = (\\sigma_{xx}+\\sigma_{yy})/2`, :math:`q = \\sqrt{((\\sigma_{xx}-
  \\sigma_{yy})/2)^2+\\sigma_{xy}^2}`).
* :func:`localization_angle` -- the Rice (1976) / Rudnicki & Rice (1975)
  shear-band bifurcation analysis: the band normal :math:`\\theta` is the
  orientation for which the acoustic tensor
  :math:`A_{ik} = n_j\\,C^{ep}_{ijkl}\\,n_l` becomes singular.  The result is
  compared with the classical closed form
  :math:`\\theta = 45^\\circ + (\\phi+\\psi)/4` (:func:`band_angle_closed_form`),
  which reduces to the Mohr-Coulomb slip plane :math:`45^\\circ+\\phi/2` for
  associated flow (``psi == phi``).

References
----------
* Rice, J. R. (1976), "The localization of plastic deformation", in
  *Theoretical and Applied Mechanics* (North-Holland), 207-220.
* Rudnicki, J. W. & Rice, J. R. (1975), *J. Mech. Phys. Solids* 23, 371-394.
* Arthur, J. R. F., Dunstan, T., Al-Ani, Q. A. J. L. & Assadi, A. (1977),
  *Geotechnique* 27(1), 53-74 (measured biaxial shear-band inclinations).
* Vardoulakis, I. & Sulem, J. (1995), *Bifurcation Analysis in Geomechanics*.

All arrays are ``numpy`` float64; every function is vectorised over the cells.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "PlaneStrainMC",
    "mean_deviatoric",
    "return_map",
    "band_angle_closed_form",
    "localization_angle",
    "biaxial_elastic_exact",
]


class PlaneStrainMC:
    """Plane-strain Mohr-Coulomb material with linear cohesion softening.

    Parameters
    ----------
    E, nu : float
        Young's modulus and Poisson ratio of the drained elastic solid.
    rho : float
        Density (only used by the explicit solver).
    phi, psi : float
        Friction and dilation angle in radians.
    c0, c_res : float
        Initial and residual cohesion (Pa).
    gp_ref : float
        Accumulated equivalent plastic strain over which the cohesion decays
        linearly from ``c0`` to ``c_res``.
    """

    def __init__(self, E=30.0e6, nu=0.3, rho=2000.0, phi=math.radians(40.0),
                 psi=math.radians(15.0), c0=350.0e3, c_res=25.0e3, gp_ref=2.0e-2):
        self.E, self.nu, self.rho = float(E), float(nu), float(rho)
        self.phi, self.psi = float(phi), float(psi)
        self.c0, self.c_res = float(c0), float(c_res)
        self.gp_ref = float(gp_ref)
        a = self.E/((1.0+self.nu)*(1.0-2.0*self.nu))
        self.Dxx = a*(1.0-self.nu)
        self.Dyy = a*(1.0-self.nu)
        self.Dxy = a*self.nu
        self.G = a*(1.0-2.0*self.nu)/2.0
        self.sin_phi = math.sin(self.phi)
        self.cos_phi = math.cos(self.phi)
        self.sin_psi = math.sin(self.psi)
        self.cp = math.sqrt(self.Dxx/self.rho)          # plane-strain P wave speed
        self.bulk2d = a                                  # dm = a * de_vol

    # -- derived quantities -------------------------------------------------
    def cohesion(self, gp):
        """Softened cohesion for the accumulated plastic strain field ``gp``."""
        gp = np.asarray(gp, dtype=float)
        if self.gp_ref <= 0.0:
            return np.full_like(gp, self.c_res)
        f = np.clip(gp/self.gp_ref, 0.0, 1.0)
        return self.c0 + (self.c_res-self.c0)*f

    def yield_function(self, sxx, syy, sxy, c):
        m, q = mean_deviatoric(sxx, syy, sxy)
        return q + m*self.sin_phi - c*self.cos_phi

    def elastic_operator(self, qx, qy, qxy):
        """Elastic operator ``C : q`` in the (xx, yy, xy) Voigt ordering."""
        return (self.Dxx*qx + self.Dxy*qy, self.Dxy*qx + self.Dyy*qy, self.G*qxy)


def mean_deviatoric(sxx, syy, sxy):
    """Tension-positive mean stress ``m`` and deviatoric radius ``q`` (2-D)."""
    sxx = np.asarray(sxx, dtype=float)
    syy = np.asarray(syy, dtype=float)
    sxy = np.asarray(sxy, dtype=float)
    return 0.5*(sxx+syy), np.sqrt(0.25*(sxx-syy)**2 + sxy**2)


def _flow_direction(mat, sxx, syy, sxy):
    """Gradient of the plastic potential (non-associated Mohr-Coulomb)."""
    s = 0.25*(sxx-syy)
    q = np.maximum(np.sqrt(s*s + sxy*sxy), 1e-30)
    return (0.5*s/q + 0.5*mat.sin_psi,
            -0.5*s/q + 0.5*mat.sin_psi,
            sxy/q)


def return_map(mat: PlaneStrainMC, sxx, syy, sxy, c):
    """Implicit return to the Mohr-Coulomb surface (apex capped).

    Returns ``(sxx, syy, sxy, dgp)`` where ``dgp`` is the increment of the
    equivalent plastic strain, ``sqrt(2/3 * de_p:de_p)``.
    """
    sxx = np.asarray(sxx, dtype=float)
    syy = np.asarray(syy, dtype=float)
    sxy = np.asarray(sxy, dtype=float)
    c = np.asarray(c, dtype=float)
    m, q = mean_deviatoric(sxx, syy, sxy)
    F = q + m*mat.sin_phi - c*mat.cos_phi
    act = F > 0.0
    dgp = np.zeros_like(sxx)
    if not act.any():
        return sxx, syy, sxy, dgp
    sx, sy, sxyo = sxx.copy(), syy.copy(), sxy.copy()
    # (1) cone tip: m >= c cot(phi) can only be projected onto the apex
    m_tip = c*mat.cos_phi/mat.sin_phi
    apex = act & (m >= m_tip)
    if apex.any():
        sx = np.where(apex, m_tip, sx)
        sy = np.where(apex, m_tip, sy)
        sxyo = np.where(apex, 0.0, sxyo)
        dgp = np.where(apex, np.abs(m_tip-m)/mat.bulk2d, dgp)
    # (2) regular cone: capped Newton on the plastic multiplier
    cone = act & (~apex)
    if cone.any():
        qx, qy, qxy = _flow_direction(mat, np.where(cone, sxx, 0.0),
                                      np.where(cone, syy, 0.0),
                                      np.where(cone, sxy, 0.0))
        cx, cy, cxy = mat.elastic_operator(qx, qy, qxy)
        den = qx*cx + qy*cy + 2.0*qxy*cxy
        den = np.where(np.abs(den) > 1e-12*mat.Dxx, den, mat.Dxx)
        dlam = np.zeros_like(sxx)
        cap = np.where(cone, 100.0*np.maximum(F, 0.0)/np.abs(den), 0.0)
        tol = 1e-8*float(c.max()) if c.size else 1e-8
        for _ in range(60):
            m2, q2 = mean_deviatoric(sxx-dlam*cx, syy-dlam*cy, sxy-dlam*cxy)
            F2 = q2 + m2*mat.sin_phi - c*mat.cos_phi
            step = np.where(cone, F2/np.abs(den), 0.0)
            dlam = np.clip(dlam+step, 0.0, cap)
            if np.max(np.abs(np.where(cone, step, 0.0))) < tol:
                break
        dg = dlam*np.sqrt((qx**2 + qy**2 + 2.0*qxy**2)*2.0/3.0)
        sx = np.where(cone, sxx-dlam*cx, sx)
        sy = np.where(cone, syy-dlam*cy, sy)
        sxyo = np.where(cone, sxy-dlam*cxy, sxyo)
        dgp = np.where(cone, dg, dgp)
    return sx, sy, sxyo, dgp


# ----------------------------------------------------------------------------
# shear-band bifurcation analysis
# ----------------------------------------------------------------------------
def band_angle_closed_form(phi, psi):
    """Literature law :math:`\\theta = 45^\\circ + (\\phi+\\psi)/4` (degrees).

    ``theta`` is measured from the direction of the minor principal stress
    (the horizontal axis of a biaxial compression specimen).  ``psi = phi``
    (associated flow) recovers the Mohr-Coulomb slip plane
    :math:`45^\\circ + \\phi/2`.
    """
    return 45.0 + (math.degrees(phi)+math.degrees(psi))/4.0


def _elastoplastic_tangent(mat: PlaneStrainMC, hardening=-1.0e6):
    """Plane-strain tangent modulus in Voigt form at a biaxial stress state.

    Compression-positive convention is used internally so that the yield
    function is the classical ``q - m sin(phi) - c cos(phi) = 0``.
    """
    a = mat.E/((1.0+mat.nu)*(1.0-2.0*mat.nu))
    D = np.array([[a*(1.0-mat.nu), a*mat.nu, 0.0],
                  [a*mat.nu, a*(1.0-mat.nu), 0.0],
                  [0.0, 0.0, a*(1.0-2.0*mat.nu)/2.0]])
    P = np.array([1.0-mat.sin_phi, -(1.0+mat.sin_phi), 0.0])
    Q = np.array([1.0-mat.sin_psi, -(1.0+mat.sin_psi), 0.0])
    DQ, PD = D @ Q, P @ D
    return D - np.outer(DQ, PD)/(hardening + P @ D @ Q)


def _voigt_to_tensor4(C):
    """Map a 3x3 Voigt matrix (xx, yy, xy) to the 2x2x2x2 tensor."""
    pairs = [(0, 0), (1, 1), (0, 1)]
    C4 = np.zeros((2, 2, 2, 2))
    for a, (i, j) in enumerate(pairs):
        for b, (k, l) in enumerate(pairs):
            ij = [(i, j)] + ([(j, i)] if i != j else [])
            kl = [(k, l)] + ([(l, k)] if k != l else [])
            for (ii, jj) in ij:
                for (kk, ll) in kl:
                    C4[ii, jj, kk, ll] += C[a, b]
    return C4


def localization_angle(mat: PlaneStrainMC, n_angles=1801, hardening=-1.0e6):
    """Critical band orientation from ``det(A(theta)) = 0`` (Rice 1976).

    Returns ``(theta_min_deg, roots_deg)`` where ``theta_min_deg`` is the angle
    (from the minor principal stress direction, in ``(0, 90)``) at which the
    determinant of the acoustic tensor is smallest, and ``roots_deg`` are the
    sign changes of that determinant (the conjugate pair).
    """
    C4 = _voigt_to_tensor4(_elastoplastic_tangent(mat, hardening))
    thetas = np.linspace(1e-6, 0.5*math.pi-1e-6, n_angles)
    dets = np.empty(n_angles)
    for i, t in enumerate(thetas):
        n = np.array([math.cos(t), math.sin(t)])
        A = np.einsum('j,ijkl,l->ik', n, C4, n)
        dets[i] = np.linalg.det(A)
    sign_change = np.where(np.sign(dets[:-1]) != np.sign(dets[1:]))[0]
    roots = np.degrees(thetas[sign_change])
    return float(np.degrees(thetas[int(np.argmin(dets))])), roots


def biaxial_elastic_exact(eps_axial, sig3, E=None, nu=None, mat=None):
    """Closed-form homogeneous plane-strain biaxial response (tension positive).

    ``eps_axial`` is the (negative in compression) axial strain in the loading
    direction; the lateral stress is prescribed as ``-sig3``.  Because the
    specimen, the material and the boundary data are homogeneous and the
    problem is linear, this is the *exact* solution of the boundary-value
    problem, so a numerical solver may be checked against it to machine
    precision (up to boundary-layer effects of the discrete load application).
    """
    if mat is not None:
        E, nu = mat.E, mat.nu
        Dxx, Dxy, Dyy = mat.Dxx, mat.Dxy, mat.Dyy
    else:
        a = E/((1.0+nu)*(1.0-2.0*nu))
        Dxx, Dxy, Dyy = a*(1.0-nu), a*nu, a*(1.0-nu)
    eps_lat = (-sig3 - Dxy*eps_axial)/Dxx      # sigma_xx = -sig3
    sig_xx = Dxx*eps_lat + Dxy*eps_axial
    sig_yy = Dxy*eps_lat + Dyy*eps_axial
    return {"eps_lateral": eps_lat, "sig_xx": sig_xx, "sig_yy": sig_yy,
            "eps_axial": eps_axial}