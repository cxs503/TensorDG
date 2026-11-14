"""Collision operators: BGK (single-relaxation) and MRT (multi-relaxation)."""

import torch
from .lattice import D2Q9


def bgk_collision(f, rho, ux, uy, omega):
    """BGK collision: f_out = f - omega * (f - feq).

    Parameters
    ----------
    f : torch.Tensor (Q, Ny, Nx)
        Pre-collision distribution.
    rho, ux, uy : torch.Tensor (Ny, Nx)
        Macroscopic fields.
    omega : float
        Relaxation parameter (1/tau).

    Returns
    -------
    f_out : torch.Tensor (Q, Ny, Nx)
        Post-collision distribution.
    """
    feq = D2Q9.equilibrium(rho, ux, uy)
    return f - omega * (f - feq)


def mrt_collision(f, rho, ux, uy, s_vec):
    """MRT collision in moment space.

    Parameters
    ----------
    f : torch.Tensor (Q, Ny, Nx)
        Pre-collision distribution.
    rho, ux, uy : torch.Tensor (Ny, Nx)
        Macroscopic fields.
    s_vec : torch.Tensor (Q,)
        Relaxation rates for each moment.

    Returns
    -------
    f_out : torch.Tensor (Q, Ny, Nx)
        Post-collision distribution.
    """
    Q = D2Q9.Q
    Ny, Nx = rho.shape

    # Transformation matrix M (moment basis)
    # Standard D2Q9 MRT ordering:
    # m = [rho, e, epsilon, jx, qx, jy, qy, pxx, pxy]^T
    M = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [-4, -1, -1, -1, -1, 2, 2, 2, 2],
        [4, -2, -2, -2, -2, 1, 1, 1, 1],
        [0, 1, 0, -1, 0, 1, -1, -1, 1],
        [0, -2, 0, 2, 0, 1, -1, -1, 1],
        [0, 0, 1, 0, -1, 1, 1, -1, -1],
        [0, 0, -2, 0, 2, 1, 1, -1, -1],
        [0, 1, -1, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 1, -1],
    ], dtype=torch.float64)

    # Inverse of M
    M_inv = torch.inverse(M)

    # Compute moments m = M @ f
    f_flat = f.view(Q, -1)  # (Q, Ny*Nx)
    m = M @ f_flat  # (Q, Ny*Nx)

    # Equilibrium moments
    feq = D2Q9.equilibrium(rho, ux, uy)
    feq_flat = feq.view(Q, -1)
    m_eq = M @ feq_flat

    # Collision in moment space: m_out = m - S @ (m - m_eq)
    S = torch.diag(s_vec)
    m_out = m - S @ (m - m_eq)

    # Transform back: f_out = M_inv @ m_out
    f_out_flat = M_inv @ m_out
    f_out = f_out_flat.view(Q, Ny, Nx)

    return f_out
