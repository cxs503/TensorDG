"""CASE-BIAXIAL-SHEAR-BAND: strain-controlled plane-strain biaxial compression.

Discretisation
--------------
Q4 bilinear quadrilaterals on a ``nx x ny`` structured grid of the specimen
``W x H`` with **full 2x2 Gauss integration** (rank sufficient, no hourglass
control), lumped nodal mass ``rho*A_trib``, stress stored at the four Gauss
points of every element, and explicit central differences in time.

Quasi-static control
--------------------
A pure velocity driven explicit relaxation cannot be quasi-static: damping the
nodal velocity towards zero also kills the uniform straining part of the
motion.  The loading is therefore imposed *kinematically* as a uniform axial
strain rate :math:`\\dot\\varepsilon_{yy} = -2 v_0/H` added to the strain-rate
field, and the nodal velocity field carries **only the fluctuation** (lateral
expansion, shear-band kinematics, elastic waves).  The fluctuation is damped by
``v <- (1-damp) v``, so the specimen relaxes to equilibrium while the axial
strain advances linearly in time; ``KE/IE`` is reported as the quasi-static
indicator.

Softening is made mesh-objective by a non-local rule: the accumulated plastic
strain is smoothed with a box kernel of radius ``n_r`` cells before it enters
the cohesion law (``ell`` = non-local length).
"""
from __future__ import annotations

import math

import numpy as np

from .plasticity_2d import PlaneStrainMC, return_map, mean_deviatoric

__all__ = ["BiaxialConfig", "run_biaxial", "band_angle_from_field",
           "structure_tensor_angle"]


class BiaxialConfig:
    """Geometry, load and numerical parameters of the biaxial experiment."""

    def __init__(self, nx=30, ny=60, W=1.0, H=2.0, sig3=50.0e3, v0=1.0e-3,
                 n_steps=9000, damping=0.1, mass_scaling=1.0e4, cfl=0.35,
                 nl_length=0.05, imperfection=(0.0, 0.5, 0.12, 0.5),
                 lateral_bc="pressure", servo_gain=0.0):
        self.nx, self.ny = int(nx), int(ny)
        self.W, self.H = float(W), float(H)
        self.sig3 = float(sig3)
        self.v0 = float(v0)
        self.n_steps = int(n_steps)
        self.damping = float(damping)
        self.mass_scaling = float(mass_scaling)
        self.cfl = float(cfl)
        # imperfection = (x_c/W, y_c/H, radius/W, cohesion factor)
        self.imperfection = imperfection
        self.nl_length = float(nl_length)
        self.lateral_bc = lateral_bc      # "pressure" or "rigid" (oedometer)
        self.servo_gain = float(servo_gain)   # servo-controlled confining pressure


_GAUSS = 1.0/math.sqrt(3.0)
_GPS = [(-_GAUSS, -_GAUSS), (_GAUSS, -_GAUSS), (_GAUSS, _GAUSS), (-_GAUSS, _GAUSS)]
_CORNERS = [(0, 0), (1, 0), (1, 1), (0, 1)]


def _shape_gradients(dx, dy):
    """dN/dx and dN/dy at the four Gauss points (node order SW, SE, NE, NW)."""
    dNdx, dNdy = [], []
    for Xi, Eta in _GPS:
        dNdxi = np.array([-(1-Eta), (1-Eta), (1+Eta), -(1+Eta)])/4.0
        dNdeta = np.array([-(1-Xi), -(1+Xi), (1+Xi), (1-Xi)])/4.0
        dNdx.append(dNdxi*2.0/dx)
        dNdy.append(dNdeta*2.0/dy)
    return dNdx, dNdy


def _blur(f, r):
    """Box blur of radius ``r`` cells in both directions (non-local average)."""
    if r < 1:
        return f
    out = f
    for axis in (0, 1):
        acc = np.zeros_like(out)
        for s in range(-r, r+1):
            acc += np.roll(out, s, axis=axis)
        out = acc/(2*r+1)
    return out


def run_biaxial(mat: PlaneStrainMC, cfg: BiaxialConfig, elastic_only=False):
    """Run the strain-controlled biaxial experiment.

    Returns a dict with the strain history, the stress response, the
    accumulated plastic strain fields (total and post-peak) and the
    quasi-static indicators.
    """
    nx, ny, W, H = cfg.nx, cfg.ny, cfg.W, cfg.H
    dx, dy = W/nx, H/ny
    dNdx, dNdy = _shape_gradients(dx, dy)
    wt = dx*dy/4.0

    cpe = mat.cp/math.sqrt(cfg.mass_scaling)
    dt = cfg.cfl*min(dx, dy)/cpe
    rate = -2.0*cfg.v0/H                    # imposed axial strain rate

    dec = np.full((nx, ny), -cfg.sig3)
    sxx = [dec.copy() for _ in range(4)]
    syy = [dec.copy() for _ in range(4)]
    sxy = [np.zeros((nx, ny)) for _ in range(4)]
    gp = np.zeros((nx, ny))
    gp_post = np.zeros((nx, ny))
    vx = np.zeros((nx+1, ny+1))
    vy = np.zeros((nx+1, ny+1))
    trib = np.zeros((nx+1, ny+1))
    for di in (0, 1):
        for dj in (0, 1):
            trib[di:di+nx, dj:dj+ny] += 0.25*dx*dy
    mass = mat.rho*cfg.mass_scaling*trib

    Xc = (np.arange(nx)+0.5)*dx
    Yc = (np.arange(ny)+0.5)*dy
    imp_x, imp_y, imp_r, imp_f = cfg.imperfection
    imp = ((Xc[:, None]-imp_x*W)**2 + (Yc[None, :]-imp_y*H)**2) < (imp_r*W)**2
    side_w = np.ones(ny+1)
    side_w[0] = side_w[-1] = 0.5
    n_r = max(0, int(round(cfg.nl_length/dx)))

    eps_hist, q_hist, m_hist, ke_hist, ie_hist = [], [], [], [], []
    eps_lat = 0.0
    peak_q, peak_i, post = -np.inf, -1, False

    for step in range(cfg.n_steps):
        c_field = np.full((nx, ny), 1.0e12) if elastic_only else mat.cohesion(gp)
        c_field = np.where(imp, imp_f*c_field, c_field)
        dgp_sum = np.zeros((nx, ny))
        exx_aff = 0.0
        if cfg.servo_gain > 0.0:
            mk = [mean_deviatoric(sxx[k], syy[k], sxy[k]) for k in range(4)]
            sig_xx_meas = float(np.mean([(m_+q_).mean() for m_, q_ in mk]))
            exx_aff = -cfg.servo_gain*(sig_xx_meas + cfg.sig3)/mat.Dxx
            eps_lat += exx_aff*dt
        for k in range(4):
            vxe = [vx[di:di+nx, dj:dj+ny] for di, dj in _CORNERS]
            vye = [vy[di:di+nx, dj:dj+ny] for di, dj in _CORNERS]
            exx = exx_aff + sum(dNdx[k][i]*vxe[i] for i in range(4))
            eyy = rate + sum(dNdy[k][i]*vye[i] for i in range(4))
            gxy = sum(dNdy[k][i]*vxe[i] + dNdx[k][i]*vye[i] for i in range(4))
            sxx[k] = sxx[k] + dt*(mat.Dxx*exx + mat.Dxy*eyy)
            syy[k] = syy[k] + dt*(mat.Dxy*exx + mat.Dyy*eyy)
            sxy[k] = sxy[k] + dt*mat.G*gxy
            sxx[k], syy[k], sxy[k], dgp = return_map(mat, sxx[k], syy[k], sxy[k], c_field)
            dgp_sum += dgp
        gp = _blur(gp + 0.25*dgp_sum, n_r)
        if post:
            gp_post = _blur(gp_post + 0.25*dgp_sum, n_r)
        # internal nodal forces  f = -B^T sigma  (momentum: m a = f_ext - f_int)
        fx = np.zeros((nx+1, ny+1))
        fy = np.zeros_like(fx)
        for k in range(4):
            for i, (di, dj) in enumerate(_CORNERS):
                fx[di:di+nx, dj:dj+ny] -= wt*(dNdx[k][i]*sxx[k] + dNdy[k][i]*sxy[k])
                fy[di:di+nx, dj:dj+ny] -= wt*(dNdy[k][i]*syy[k] + dNdx[k][i]*sxy[k])
        if cfg.lateral_bc == "pressure":
            # confining pressure on the lateral faces (tension positive -> inward)
            fx[0, 1:-1] += cfg.sig3*dy*side_w[1:-1]
            fx[-1, 1:-1] -= cfg.sig3*dy*side_w[1:-1]
        vx += dt*fx/mass
        vy += dt*fy/mass
        vx *= (1.0-cfg.damping)
        vy *= (1.0-cfg.damping)
        vy[0, :] = 0.0       # fluctuation only: the affine part is in ``rate``
        vy[-1, :] = 0.0
        if cfg.lateral_bc == "rigid":
            vx[:] = 0.0        # rigid box: eps_xx == 0 everywhere (oedometer)
        if step % 50 == 0:
            mm = sum(0.25*mean_deviatoric(sxx[k], syy[k], sxy[k])[0] for k in range(4))
            qq = sum(0.25*mean_deviatoric(sxx[k], syy[k], sxy[k])[1] for k in range(4))
            ke = 0.5*float((mass*(vx**2+vy**2)).sum())
            ie = 0.25*float(sum((sxx[k]*sxx[k]/(2*mat.Dxx)).sum() for k in range(4)))/4.0
            eps_hist.append((step+1)*dt*abs(rate))
            q_hist.append(float(qq.mean()))
            m_hist.append(float(mm.mean()))
            ke_hist.append(ke)
            ie_hist.append(ie)
            if q_hist[-1] > peak_q:
                peak_q, peak_i = q_hist[-1], len(q_hist)-1
            elif (not post) and q_hist[-1] < 0.95*peak_q:
                post = True     # peak passed: start accumulating the band field
    q = sum(0.25*mean_deviatoric(sxx[k], syy[k], sxy[k])[1] for k in range(4))
    m = sum(0.25*mean_deviatoric(sxx[k], syy[k], sxy[k])[0] for k in range(4))
    return {
        "eps": np.array(eps_hist), "q_mean": np.array(q_hist),
        "m_mean": np.array(m_hist), "ke": np.array(ke_hist),
        "ie": np.array(ie_hist),
        "q_final": q, "m_final": m, "gp": gp, "gp_post": gp_post,
        "post_peak": post, "dt": dt, "rate": rate, "eps_lateral": eps_lat,
        "x": Xc, "y": Yc, "W": W, "H": H, "nx": nx, "ny": ny,
        "n_r": n_r, "ke_over_ie": float(np.median(np.array(ke_hist) /
                                                 np.maximum(np.array(ie_hist), 1e-30))),
        "q_peak": float(peak_q),
        "eps_peak": float(eps_hist[peak_i]) if peak_i >= 0 else float("nan"),
    }


def band_angle_from_field(field, x, y, threshold=0.5, min_contrast=1.5):
    """Orientation of a localized band from its plastic-strain field (degrees).

    The centroid and the principal axis of the covariance of the field values
    above ``threshold * max(field)`` give the band direction; the angle is
    measured from the horizontal axis (the minor-principal-stress direction of
    a biaxial compression test) and folded into ``(0, 90]``.
    Returns ``(angle_deg, n_points, contrast)``; ``angle`` is ``nan`` when the
    field carries no localized structure.
    """
    field = np.asarray(field, dtype=float)
    fmax = float(field.max()) if field.size else 0.0
    if not np.isfinite(fmax) or fmax <= 0.0:
        return float("nan"), 0, float("nan")
    contrast = float(fmax/max(field.mean(), 1e-30))
    mask = field > threshold*fmax
    npts = int(mask.sum())
    if npts < 20 or contrast < min_contrast:
        # no localized structure: an orientation would be meaningless
        return float("nan"), npts, contrast
    Xg = np.broadcast_to(np.asarray(x)[:, None], field.shape)[mask]
    Yg = np.broadcast_to(np.asarray(y)[None, :], field.shape)[mask]
    w = field[mask]
    Xm, Ym = (Xg*w).sum()/w.sum(), (Yg*w).sum()/w.sum()
    Cxx = ((Xg-Xm)**2*w).sum()/w.sum()
    Cyy = ((Yg-Ym)**2*w).sum()/w.sum()
    Cxy = ((Xg-Xm)*(Yg-Ym)*w).sum()/w.sum()
    evals, evecs = np.linalg.eigh(np.array([[Cxx, Cxy], [Cxy, Cyy]]))
    d = evecs[:, int(np.argmax(evals))]
    ang = math.degrees(math.atan2(abs(d[1]), abs(d[0])))
    return float(min(ang, 90.0)), npts, contrast


def structure_tensor_angle(field, dx, dy):
    """Independent estimate of the band orientation via the structure tensor.

    For a banded scalar field the eigenvector of the structure tensor
    associated with the *smallest* eigenvalue is parallel to the band, so this
    is a second, gradient-based estimator of the same angle.
    """
    field = np.asarray(field, dtype=float)
    if not np.isfinite(field).all() or field.max() <= 0.0:
        return float("nan")
    gx, gy = np.gradient(field, dx, dy)
    Jxx, Jyy, Jxy = gx*gx, gy*gy, gx*gy
    evals, evecs = np.linalg.eigh(np.array([[Jxx.sum(), Jxy.sum()],
                                            [Jxy.sum(), Jyy.sum()]]))
    d = evecs[:, int(np.argmin(evals))]
    return float(min(math.degrees(math.atan2(abs(d[1]), abs(d[0]))), 90.0))