"""Localise the instability of tensordg.euler_curved (scratch, not shipped).

Reimplements ``CurvilinearEuler.rhs`` with switches so each term can be tested
independently against the linear spectrum about a uniform flow.
"""
import numpy as np

from tensordg.euler_curved import (
    CurvilinearEuler,
    euler_flux,
    conserved_from_primitive,
    hllc_normal,
    straight_box_mesh,
    warped_box_mesh,
)


def rhs_variant(self, state, face_sign=-1.0, lo_sign=1.0, volume_sign=-1.0,
                flux="hllc", volume=True, face=True):
    basis, weights = self.basis, self.w
    sampled = np.einsum("qk,nklc,rl->nqrc", basis, state, basis)
    flux_x, flux_y = euler_flux(sampled, self.gamma)
    contravariant_xi = (self.metric_xi[..., 0, None] * flux_x
                        + self.metric_xi[..., 1, None] * flux_y)
    contravariant_eta = (self.metric_eta[..., 0, None] * flux_x
                         + self.metric_eta[..., 1, None] * flux_y)
    divergence = (self.differentiate(contravariant_xi, 0)
                  + self.differentiate(contravariant_eta, 1))
    volume_term = np.einsum("qk,rl,nqrc,q,r->nklc", basis, basis, divergence,
                            weights, weights) if volume else np.zeros_like(state)

    traces = self.traces(state)
    face_term = np.zeros_like(state)
    for axis in (0, 1):
        for side in (-1, 1):
            geometry = self.face[(axis, side)]
            inner, outer = self._face_states(traces, axis, side)
            if side > 0:
                left, right = inner, outer
            else:
                left, right = outer, inner
            if flux == "hllc":
                numerical = geometry["magnitude"][..., None] * hllc_normal(
                    left, right, geometry["unit_x"], geometry["unit_y"], self.gamma)
            else:  # plain rusanov in the face-normal direction
                fx_l, fy_l = euler_flux(left, self.gamma)
                fx_r, fy_r = euler_flux(right, self.gamma)
                f_l = fx_l * geometry["unit_x"][..., None] + fy_l * geometry["unit_y"][..., None]
                f_r = fx_r * geometry["unit_x"][..., None] + fy_r * geometry["unit_y"][..., None]
                v_l = (left[..., 1] * geometry["unit_x"] + left[..., 2] * geometry["unit_y"]) / left[..., 0]
                v_r = (right[..., 1] * geometry["unit_x"] + right[..., 2] * geometry["unit_y"]) / right[..., 0]
                a_l = np.sqrt(self.gamma * np.maximum(
                    (self.gamma - 1) * (left[..., 3] - 0.5 * (left[..., 1] ** 2 + left[..., 2] ** 2) / left[..., 0]), 1e-12) / left[..., 0])
                a_r = np.sqrt(self.gamma * np.maximum(
                    (self.gamma - 1) * (right[..., 3] - 0.5 * (right[..., 1] ** 2 + right[..., 2] ** 2) / right[..., 0]), 1e-12) / right[..., 0])
                alpha = np.maximum(np.abs(v_l) + a_l, np.abs(v_r) + a_r)
                numerical = geometry["magnitude"][..., None] * (
                    0.5 * (f_l + f_r) - 0.5 * alpha[..., None] * (right - left))
            flux_x_in, flux_y_in = euler_flux(inner, self.gamma)
            physical = (geometry["metric"][..., 0, None] * flux_x_in
                        + geometry["metric"][..., 1, None] * flux_y_in)
            correction = numerical - physical
            if not face:
                continue
            edge_basis = np.ones(self.K) if side > 0 else lo_sign * self.signs
            if axis == 0:
                face_term += np.einsum("k,rl,nrc,r->nklc", edge_basis, basis,
                                       correction, weights)
            else:
                face_term += np.einsum("l,qk,nqc,q->nklc", edge_basis, basis,
                                       correction, weights)
    return self.apply_mass_inverse(volume_sign * volume_term + face_sign * face_term)


def spectrum(solver, variant=None, eps=1e-8, label="", **kw):
    base = np.zeros((solver.mesh.ncells, solver.K, solver.K, 4))
    base[:, 0, 0, :] = conserved_from_primitive(1.0, 0.3, -0.2, 1.0)
    func = solver.rhs if variant is None else (lambda s: rhs_variant(solver, s, **kw))
    n = base.size
    f0 = func(base).reshape(-1)
    J = np.zeros((n, n))
    flat = base.reshape(-1)
    for i in range(n):
        pert = flat.copy()
        pert[i] += eps
        J[:, i] = (func(pert.reshape(base.shape)).reshape(-1) - f0) / eps
    ev = np.linalg.eigvals(J)
    re = np.sort(ev.real)[::-1]
    print(f"[{label}] dof={n} f0={np.max(np.abs(f0)):.2e} maxRe={re[0]:+.4e} "
          f"nPos={int((re > 0).sum())} top4={np.array2string(re[:4], precision=3)}",
          flush=True)
    return J, ev


if __name__ == "__main__":
    L = 10.0
    for cells, deg, name in ((2, 1, "straight2"), (2, 3, "warped2")):
        mesh = (straight_box_mesh(cells, cells, 1, length=L) if deg == 1
                else warped_box_mesh(cells, cells, 3, length=L, amplitude=0.06))
        for p in (0, 1):
            s = CurvilinearEuler(mesh, degree=p, cfl=0.2)
            spectrum(s, label=f"{name} p={p} BASE")
            spectrum(s, label=f"{name} p={p} face_neg", variant=rhs_variant, face_sign=+1.0)
            spectrum(s, label=f"{name} p={p} lo_neg", variant=rhs_variant, lo_sign=-1.0)
            spectrum(s, label=f"{name} p={p} rusanov", variant=rhs_variant, flux="rusanov")
            spectrum(s, label=f"{name} p={p} no_vol", variant=rhs_variant, volume=False)
            spectrum(s, label=f"{name} p={p} no_face", variant=rhs_variant, face=False)