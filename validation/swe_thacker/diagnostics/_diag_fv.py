"""Scratch diagnostic 12: independent FV reference solution vs analytic solution."""
import math
import numpy as np

from tensordg import swe_exact

G, H0, AMP, L = 9.81, 1.0, 0.5, 1.0


def fv_lf(ncell=2000, t_end=0.5, cfl=0.2, exact_bc=True):
    dx = 2 * L / ncell
    xf = -L + np.arange(ncell + 1) * dx
    xc = 0.5 * (xf[:-1] + xf[1:])
    zb = swe_exact.thacker_parabolic_bed(xc, H0, L)
    zbf = swe_exact.thacker_parabolic_bed(xf, H0, L)
    h, u = swe_exact.thacker_planar_basin_initial(xc, H0, AMP, L)
    U = np.stack([h, h * u], -1)
    t = 0.0
    steps = 0
    while t < t_end:
        c = np.sqrt(G * np.maximum(U[:, 0], 1e-12))
        smax = np.max(np.abs(U[:, 1] / np.maximum(U[:, 0], 1e-12)) + c)
        dt = min(cfl * dx / smax, t_end - t)
        F = np.stack([U[:, 1], U[:, 1] ** 2 / np.maximum(U[:, 0], 1e-12) + 0.5 * G * U[:, 0] ** 2], -1)
        # Rusanov flux at interior faces
        Ul, Ur = U[:-1], U[1:]
        Fl, Fr = F[:-1], F[1:]
        cl = np.sqrt(G * np.maximum(Ul[:, 0], 1e-12))
        cr = np.sqrt(G * np.maximum(Ur[:, 0], 1e-12))
        al = np.abs(Ul[:, 1] / np.maximum(Ul[:, 0], 1e-12)) + cl
        ar = np.abs(Ur[:, 1] / np.maximum(Ur[:, 0], 1e-12)) + cr
        a = np.maximum(al, ar)[:, None]
        Fface = 0.5 * (Fl + Fr) - 0.5 * a * (Ur - Ul)
        if exact_bc:
            fl, fr = swe_exact.thacker_planar_basin_boundary_flux(t, H0, AMP, L, G)
            FL, FR = np.array(fl), np.array(fr)
            Fface = np.vstack([FL[None, :], Fface, FR[None, :]])
        else:
            Fface = np.vstack([Fl[:1], Fface, Fr[-1:]])
        # source term (bed slope, centred)
        S = np.zeros_like(U)
        S[:, 1] = -G * U[:, 0] * np.gradient(zb, dx)
        U = U - (dt / dx) * (Fface[1:] - Fface[:-1]) + dt * S
        U[:, 0] = np.maximum(U[:, 0], 1e-10)
        t += dt
        steps += 1
    return xc, U, steps


if __name__ == "__main__":
    xc, U, steps = fv_lf(exact_bc=True)
    h_e, u_e, eta_e = swe_exact.thacker_planar_basin(xc, 0.5, H0, AMP, L, G)
    u = U[:, 1] / U[:, 0]
    eta = h = U[:, 0]
    eta_num = h + swe_exact.thacker_parabolic_bed(xc, H0, L)
    print(f"FV-Rusanov N=2000 steps={steps} (exact BC)")
    print(f"  rel L2 eta error = {np.sqrt(np.mean((eta_num-eta_e)**2))/np.abs(eta_e).max()*100:.4f}%")
    print(f"  rel L2 u   error = {np.sqrt(np.mean((u-u_e)**2))/np.abs(u_e).max()*100:.4f}%")
    print(f"  u range numeric  = [{u.min():.4f},{u.max():.4f}]  exact = {u_e[0]:.4f}")
    print(f"  volume numeric   = {h.sum()*2*L/len(h):.6f}  exact = "
          f"{2*(H0+AMP**2/(4*H0)*math.sin(math.sqrt(2*G*H0)/L*0.5)**2)+4/3*H0*L:.6f}")