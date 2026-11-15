import numpy as np, time
from tensordg.euler_curved import (CurvilinearEuler, straight_box_mesh,
                                  warped_box_mesh, conserved_from_primitive)
from tensordg.euler_2d import vortex_primitive
import torch

BOX = 10.0
G = 3


def spectrum(mesh, p=1, eps=1e-8, label=""):
    s = CurvilinearEuler(mesh, degree=p, cfl=0.2)
    base = np.zeros((mesh.ncells, s.K, s.K, 4))
    base[:, 0, 0, :] = conserved_from_primitive(1.0, 0.3, -0.2, 1.0)
    n = base.size
    f0 = s.rhs(base).reshape(-1)
    J = np.zeros((n, n))
    flat = base.reshape(-1)
    for i in range(n):
        pert = flat.copy()
        pert[i] += eps
        J[:, i] = (s.rhs(pert.reshape(base.shape)).reshape(-1) - f0) / eps
    ev = np.linalg.eigvals(J)
    re = np.sort(ev.real)[::-1]
    print(f"[spectrum {label}] cells={mesh.ncells} p={p} nq={s.nq} dof={n} "
          f"maxRe={re[0]:+.4e} nPos={int((re > 0).sum())} "
          f"top5={np.array2string(re[:5], precision=3)}", flush=True)
    return re


def conv(mesh_builder, cells_list, p, cfl, t=0.1, strength=1.0, label=""):
    def exact(time):
        def prim(x, y):
            r, u, v, pp = vortex_primitive(
                torch.as_tensor(np.ascontiguousarray(x)),
                torch.as_tensor(np.ascontiguousarray(y)),
                time=time, strength=strength, background_velocity=(1.0, 1.0), period=BOX)
            return (r.numpy(), u.numpy(), v.numpy(), pp.numpy())
        return prim
    errs = []
    for c in cells_list:
        s = CurvilinearEuler(mesh_builder(c), degree=p, cfl=cfl)
        st = s.project(exact(0.0))
        st, dt, steps = s.evolve(st, t)
        e = s.relative_errors(st, exact(t))["worst"]
        errs.append(e)
        print(f"[conv {label}] cells={c} cfl={cfl} err={e:.4e} dt={dt:.3e} steps={steps}", flush=True)
    orders = [np.log(errs[i] / errs[i + 1]) / np.log(2.0) for i in range(len(errs) - 1)]
    print(f"[conv {label}] orders={np.array2string(np.array(orders), precision=3)}", flush=True)
    return errs


if __name__ == "__main__":
    t0 = time.time()
    spectrum(straight_box_mesh(2, 2, 1, length=BOX), label="straight2")
    spectrum(straight_box_mesh(4, 4, 1, length=BOX), label="straight4")
    spectrum(warped_box_mesh(2, 2, G, length=BOX, amplitude=0.06), label="warped2")
    print("t", time.time() - t0, flush=True)
    conv(lambda c: straight_box_mesh(c, c, 1, length=BOX), [8, 16, 32], 1, 0.2, label="straight")
    conv(lambda c: warped_box_mesh(c, c, G, length=BOX, amplitude=0.06), [8, 16, 32], 1, 0.2, label="warped")
    conv(lambda c: straight_box_mesh(c, c, 1, length=BOX), [8, 16, 32], 1, 0.02, label="straight-cfl.02")
    print("total", time.time() - t0, flush=True)