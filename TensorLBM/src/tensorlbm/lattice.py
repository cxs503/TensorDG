"""D2Q9 lattice definition — velocities, weights, and opposite indices."""

import torch


class D2Q9:
    """D2Q9 lattice: 9 velocities, standard weights, opposite index map."""

    Q = 9

    # Velocity vectors (e_x, e_y)
    e = torch.tensor([
        [0, 0],   # 0: rest
        [1, 0],   # 1: east
        [0, 1],   # 2: north
        [-1, 0],  # 3: west
        [0, -1],  # 4: south
        [1, 1],   # 5: northeast
        [-1, 1],  # 6: northwest
        [-1, -1], # 7: southwest
        [1, -1],  # 8: southeast
    ], dtype=torch.float64)

    # Weights
    w = torch.tensor([
        4 / 9,    # 0
        1 / 9,    # 1
        1 / 9,    # 2
        1 / 9,    # 3
        1 / 9,    # 4
        1 / 36,   # 5
        1 / 36,   # 6
        1 / 36,   # 7
        1 / 36,   # 8
    ], dtype=torch.float64)

    # Opposite index map: opp[i] = j means e_j = -e_i
    opp = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.long)

    # Speed of sound squared (lattice units)
    cs2 = torch.tensor(1 / 3, dtype=torch.float64)

    @classmethod
    def equilibrium(cls, rho, ux, uy):
        """Compute equilibrium distribution f^eq from macroscopic fields.

        Parameters
        ----------
        rho : torch.Tensor (Ny, Nx)
            Density field.
        ux, uy : torch.Tensor (Ny, Nx)
            Velocity components.

        Returns
        -------
        feq : torch.Tensor (Q, Ny, Nx)
            Equilibrium distribution functions.
        """
        Q = cls.Q
        Ny, Nx = rho.shape
        feq = torch.zeros(Q, Ny, Nx, dtype=rho.dtype, device=rho.device)

        u_sq = ux ** 2 + uy ** 2
        for i in range(Q):
            eu = cls.e[i, 0] * ux + cls.e[i, 1] * uy
            feq[i] = rho * cls.w[i] * (
                1.0 + 3.0 * eu + 4.5 * eu ** 2 - 1.5 * u_sq
            )
        return feq

    @classmethod
    def macroscopic(cls, f):
        """Compute macroscopic density and velocity from distribution f.

        Parameters
        ----------
        f : torch.Tensor (Q, Ny, Nx)
            Distribution functions.

        Returns
        -------
        rho : torch.Tensor (Ny, Nx)
            Density.
        ux, uy : torch.Tensor (Ny, Nx)
            Velocity components.
        """
        rho = f.sum(dim=0)
        ux = (cls.e[:, 0].view(-1, 1, 1) * f).sum(dim=0) / rho
        uy = (cls.e[:, 1].view(-1, 1, 1) * f).sum(dim=0) / rho
        return rho, ux, uy
