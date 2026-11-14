"""Streaming operator with periodic boundary conditions."""

import torch
from .lattice import D2Q9


def stream_periodic(f):
    """Stream distribution functions with periodic BCs.

    Parameters
    ----------
    f : torch.Tensor (Q, Ny, Nx)
        Post-collision distribution.

    Returns
    -------
    f_streamed : torch.Tensor (Q, Ny, Nx)
        Streamed distribution.
    """
    Q, Ny, Nx = f.shape
    f_out = torch.empty_like(f)

    for i in range(Q):
        ex = int(D2Q9.e[i, 0])
        ey = int(D2Q9.e[i, 1])
        # Roll along each spatial dimension
        f_out[i] = torch.roll(f[i], shifts=(ey, ex), dims=(0, 1))

    return f_out
