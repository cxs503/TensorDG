"""Sign-combination scan for the face term of tensordg.euler_curved (scratch)."""
import numpy as np

from tensordg.euler_curved import (
    CurvilinearEuler, euler_flux, conserved_from_primitive, hllc_normal,
    straight_box_mesh, warped_box_mesh,
)
from scratch.variants import rhs_variant  # noqa: F401  (reuse the replication)


def rhs_signed(self, state, hi_sign=1.0, lo_sign=1.0):
    basis, weights = self.basis, self.w
    sampled = np.einsum("qk,nklc,rl->nqrc", basis, state, basis)
    flux_x, flux_y = euler_flux(sampled, self.gamma)
    c_xi = (self.metric_xi[..., 0, None] * flux_x
            + self.metric_xi[..., 1, None] * flux_y)
    c_eta = (self.metric_eta[..., 0, None] * flux_x
             + self.metric_eta[..., 1, None] * flux_y)
    divergence = self.differentiate(c_xi, 0) + self.differentiate(c_eta, 1)
    volume = np.einsum("qk,rl,nqrc,q,r->nklc", basis, basis, divergence,
                       weights, weights)
    traces = self.traces(state)
    face = np.zeros_like(state)
    for axis in (0, 1):
        for side in (-1, 1):
            geometry = self.face[(axis, side)]
            inner, outer = self._face_states(traces, axis, side)
            left, right = (inner, outer) if side > 0 else (outer, inner)
            numerical = geometry["magnitude"][..., None] * hllc_normal(
                left, right, geometry["unit_x"], geometry["unit_y"], self.gamma)
            fx_in, fy_in = euler_flux(inner, self.gamma)
            physical = (geometry["metric"][..., 0, None] * fx_in
                        + geometry["metric"][..., 1, None] * fy_in)
            correction = (hi_sign if side > 0 else lo_sign) * (numerical - physical)
            edge_basis = np.ones(self.K) if side > 0 else self.signs
            if axis == 0:
                face += np.einsum("k,rl,nrc,r->nklc", edge_basis, basis,
                                  correction, weights)
            else:
                face += np.einsum("l,qk,nqc,q->nklc", edge_basis, basis,
                                  correction, weights)
    return self.apply_mass_inverse(-volume - face)


def spectrum(solver, hi_sign, lo_sign, eps=1e-8, label=""):
    base = np.zeros((solver.mesh.ncells, solver.K, solver.K, 4))
    base[:, 0, 0, :] = conserved_from_primitive(1.0, 0.3, -0.2, 1.0)
    func = lambda s: rhs_signed(solver, s, hi_sign, lo_sign)  # noqa: E731
    n = base.size
    f0 = func(base).reshape(-1)
    J = np.zeros((n, n))
    flat = base.reshape(-1)
    for i in range(n):
        pert = flat.copy()
        pert[i] += eps
        J[:, i] = (func(pert.reshape(base.shape)).reshape(-1) - f0) / eps
    re = np.sort(np.linalg.eigvals(J).real)[::-1]
    print(f"[{label} hi={hi_sign:+.0f} lo={lo_sign:+.0f}] maxRe={re[0]:+.3e} "
          f"nPos={int((re > 0).sum())}", flush=True)
    return re


if __name__ == "__main__":
    L = 10.0
    for cells, name, mesh in (
        (2, "straight2", straight_box_mesh(2, 2, 1, length=L)),
        (2, "warped2", warped_box_mesh(2, 2, 3, length=L, amplitude=0.06)),
    ):
        for p in (0, 1):
            s = CurvilinearEuler(mesh, degree=p, cfl=0.2)
            for hi, lo in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                spectrum(s, hi, lo, label=f"{name} p={p}")