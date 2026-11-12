"""Exploratory probe: spectral radius of the SIPG operator M^-1 K (1D and 2D)."""
import torch
from tensordg.heat_1d import assemble_sipg_1d
from tensordg import sipg_sparse


def spec_1d(degree, ncells, penalty):
    d = assemble_sipg_1d(ncells=ncells, degree=degree, penalty=penalty)
    M, K = d["mass_matrix"], d["diffusion"]
    s = torch.diag(M).clamp_min(0).sqrt()
    A = K / s[:, None] / s[None, :]
    w = torch.linalg.eigvalsh(0.5 * (A + A.T))
    return float(w.max()), float(w.min()), d["h"]


def spec_2d(nx, ny, degree, penalty):
    K, ops = sipg_sparse.assemble_poisson_2d_sparse(
        nx, ny, degree, penalty, torch.device("cpu"), torch.float64)
    modes = degree + 1
    hx, hy = 1.0 / nx, 1.0 / ny
    k = torch.arange(modes, dtype=torch.float64)
    # diagonal modal mass in 2D: hx*hy/4 * 2/(2k+1) * 2/(2l+1)
    mk = hx * hy * 0.5 / (2 * k + 1)
    mdiag = torch.outer(mk, mk).reshape(-1).numpy()
    import numpy as np
    from scipy.sparse import diags
    from scipy.sparse.linalg import eigsh
    s = np.sqrt(mdiag)
    Dinv = diags(1.0 / s)
    A = Dinv @ K @ Dinv
    A = ((A + A.T) * 0.5).tocsc()
    wmax = eigsh(A, k=1, which="LA", return_eigenvectors=False, tol=1e-12)[0]
    wmin = eigsh(A, k=1, which="SA", return_eigenvectors=False, tol=1e-12)[0]
    return float(wmax), float(wmin)


print("1D (alpha = penalty*(p+1)^2 ; c = lam_max*h^2/alpha)")
for penalty in (20.0, 2.0):
    for p in (1, 2, 3, 4, 5):
        row = []
        for n in (4, 8, 16, 32):
            lm, lmin, h = spec_1d(p, n, penalty)
            alpha = penalty * (p + 1) ** 2
            row.append((n, lm * h * h / alpha, lmin * h * h / alpha))
        print(f"  pen={penalty} p={p}: " + " ".join(
            f"n={n}:c={c:.3f},min={m:.2e}" for n, c, m in row))

print("\n2D (alpha = penalty*(p+1)^2 ; c = lam_max*h^2/alpha)")
for penalty in (2.0, 10.0):
    for p in (0, 1, 2, 3):
        row = []
        for n in (4, 8, 12):
            lm, lmin = spec_2d(n, n, p, penalty)
            alpha = penalty * (p + 1) ** 2
            h = 1.0 / n
            row.append((n, lm * h * h / alpha, lmin * h * h / alpha))
        print(f"  pen={penalty} p={p}: " + " ".join(
            f"n={n}:c={c:.3f},min={m:.2e}" for n, c, m in row))