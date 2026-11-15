"""Tests for the sparse list-based AOS LBM (lbmpy sparse-list-based-aos-lbm).

Covered invariants
------------------
* layout: AOS state is ``(n_fluid, Q)`` with unit stride in the direction axis;
  SoA is ``(Q, n_fluid)``; both hold exactly ``n_fluid*Q`` doubles.
* sparse domain: the pull-streaming source table matches a brute-force dense
  neighbour enumeration, and wall directions are exactly the non-fluid ones.
* equivalence: on a fully periodic domain the sparse solver reproduces the
  dense ``(Q, Ny, Nx)`` solver of :mod:`tensorlbm.streaming` (round-off level).
* conservation: bounce-back walls conserve total mass exactly.
* physics: body-force driven channel flow converges to the Poiseuille profile
  of the half-way bounce-back wall position, rel-L2 well below the 15 %
  acceptance bound of the anchor feature-ledger entry.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensorlbm import D2Q9, bgk_collision, stream_periodic
from tensorlbm.sparse_aos import (
    SparseAOSLBM,
    build_sparse_domain,
    channel_mask,
)

DTYPE = torch.float64


# --------------------------------------------------------------------- helpers
def taylor_green(ny, nx, amp=0.04):
    """Taylor-Green initial fields on a fully periodic box."""
    y = torch.arange(ny, dtype=DTYPE)
    x = torch.arange(nx, dtype=DTYPE)
    kx = 2 * torch.pi / nx
    ky = 2 * torch.pi / ny
    ux = amp * torch.sin(kx * x)[None, :] * torch.cos(ky * y)[:, None]
    uy = -amp * torch.cos(kx * x)[None, :] * torch.sin(ky * y)[:, None]
    rho = torch.ones(ny, nx, dtype=DTYPE)
    return rho, ux, uy


def run_dense(ny, nx, steps, omega=1.9):
    rho, ux, uy = taylor_green(ny, nx)
    f = D2Q9.equilibrium(rho, ux, uy).to(DTYPE)
    for _ in range(steps):
        r, uxx, uyy = D2Q9.macroscopic(f)
        f = stream_periodic(bgk_collision(f, r, uxx, uyy, omega))
    return f


def run_sparse(ny, nx, steps, omega=1.9, layout="aos"):
    mask = torch.ones(ny, nx, dtype=torch.bool)
    dom = build_sparse_domain(mask, periodic=(True, True))
    solver = SparseAOSLBM(dom, omega=omega, layout=layout, dtype=DTYPE)
    rho, ux, uy = taylor_green(ny, nx)
    solver.initialize(rho, ux, uy)
    for _ in range(steps):
        solver.step()
    return solver


# ----------------------------------------------------------------------- tests
class TestLayout:
    def test_aos_is_array_of_structures(self):
        ny, nx = 8, 12
        mask = channel_mask(ny, nx)
        dom = build_sparse_domain(mask, periodic=(True, True))
        n = dom.n_fluid
        assert n == (ny - 2) * nx

        aos = SparseAOSLBM(dom, layout="aos", dtype=DTYPE)
        soa = SparseAOSLBM(dom, layout="soa", dtype=DTYPE)

        assert aos.f.shape == (n, D2Q9.Q)
        assert aos.f.stride(1) == 1  # directions contiguous -> AOS
        assert soa.f.shape == (D2Q9.Q, n)

        # same number of doubles in both layouts, sparse vs dense
        assert aos.state_bytes == soa.state_bytes == n * D2Q9.Q * 8
        assert aos.state_bytes < D2Q9.Q * ny * nx * 8  # genuinely sparse

    def test_sparse_source_table_matches_dense_neighbours(self):
        ny, nx = 9, 11
        mask = torch.ones(ny, nx, dtype=torch.bool)
        mask[3, 4:7] = False  # interior obstacle
        mask[0, :] = False    # and a solid top row
        dom = build_sparse_domain(mask, periodic=(False, True))  # periodic in x only
        coords = dom.coords
        pos = {(int(coords[c, 0]), int(coords[c, 1])): c for c in range(dom.n_fluid)}
        for q in range(D2Q9.Q):
            ex, ey = int(D2Q9.e[q, 0]), int(D2Q9.e[q, 1])
            for c in range(dom.n_fluid):
                y = int(coords[c, 0])
                x = int(coords[c, 1])
                sy, sx = y - ey, x - ex
                sx %= nx                      # periodic in x only
                expect = pos.get((sy, sx), -1) if 0 <= sy < ny else -1
                assert int(dom.src[q, c]) == expect, (q, c, sy, sx)

    def test_rejects_empty_domain(self):
        with pytest.raises(ValueError):
            build_sparse_domain(torch.zeros(4, 4, dtype=torch.bool))


class TestEquivalenceWithDense:
    @pytest.mark.parametrize("layout", ["aos", "soa"])
    def test_matches_dense_periodic_solver(self, layout):
        ny, nx, steps = 16, 16, 80
        f_dense = run_dense(ny, nx, steps)
        solver = run_sparse(ny, nx, steps, layout=layout)
        f_sparse = solver.to_dense()
        assert torch.isfinite(f_sparse).all()
        max_diff = (f_sparse - f_dense).abs().max().item()
        scale = f_dense.abs().max().item()
        assert max_diff / scale < 1e-12, max_diff / scale

    def test_aos_and_soa_bitwise_agree(self):
        ny, nx, steps = 16, 16, 40
        a = run_sparse(ny, nx, steps, layout="aos").to_dense()
        b = run_sparse(ny, nx, steps, layout="soa").to_dense()
        assert (a - b).abs().max().item() < 1e-14

    def test_gather_scatter_roundtrip(self):
        mask = channel_mask(8, 10)
        dom = build_sparse_domain(mask, periodic=(True, True))
        field = torch.arange(80, dtype=DTYPE).reshape(8, 10)
        back = dom.scatter(dom.gather(field))
        fluid = mask
        assert torch.allclose(back[fluid], field[fluid])
        assert torch.isnan(back[~fluid]).all()


class TestConservation:
    def test_mass_conserved_with_bounce_back_walls(self):
        ny, nx = 24, 24
        dom = build_sparse_domain(channel_mask(ny, nx), periodic=(True, True))
        solver = SparseAOSLBM(dom, omega=1.8, dtype=DTYPE)
        solver.initialize(
            torch.ones(ny, nx, dtype=DTYPE),
            torch.zeros(ny, nx, dtype=DTYPE),
            torch.zeros(ny, nx, dtype=DTYPE),
        )
        m0 = solver.total_mass().item()
        for _ in range(400):
            solver.step(force=(1e-5, 0.0))
        m1 = solver.total_mass().item()
        assert abs(m1 - m0) / m0 < 1e-13


class TestTaylorGreen:
    """Viscous decay of the single Fourier mode on the sparse periodic domain."""

    def test_decay_rate_matches_vorticity_diffusion(self):
        import math

        ny = nx = 32
        omega, amp, steps = 1.9, 0.04, 2000
        y = torch.arange(ny, dtype=DTYPE)
        x = torch.arange(nx, dtype=DTYPE)
        kx, ky = 2 * math.pi / nx, 2 * math.pi / ny
        ux0 = amp * torch.sin(kx * x)[None, :] * torch.cos(ky * y)[:, None]
        uy0 = -amp * torch.cos(kx * x)[None, :] * torch.sin(ky * y)[:, None]
        rho0 = torch.ones(ny, nx, dtype=DTYPE)
        dom = build_sparse_domain(
            torch.ones(ny, nx, dtype=torch.bool), periodic=(True, True)
        )
        solver = SparseAOSLBM(dom, omega=omega, dtype=DTYPE)
        solver.initialize(rho0, ux0, uy0)
        basis = torch.sin(kx * x)[None, :] * torch.cos(ky * y)[:, None]
        steps_seen, amps = [], []
        for it in range(steps + 1):
            if it % 200 == 0:
                _, ux, _ = solver.macroscopic()
                steps_seen.append(it)
                amps.append(float(4.0 * (dom.scatter(ux) * basis).mean()))
            solver.step()
        import math as _m
        ln = [_m.log(a) for a in amps]
        n = len(steps_seen)
        ms, ml = sum(steps_seen) / n, sum(ln) / n
        rate = -sum((s - ms) * (l - ml) for s, l in zip(steps_seen, ln)) / sum(
            (s - ms) ** 2 for s in steps_seen
        )
        nu = (1.0 / omega - 0.5) / 3.0
        rate_ref = nu * (kx * kx + ky * ky)
        rel = abs(rate - rate_ref) / rate_ref
        print(f"\nTG decay rate {rate:.6e} vs {rate_ref:.6e} -> {100*rel:.3f}%")
        assert rel < 0.03


class TestPoiseuille:
    """Body-force channel flow vs the half-way bounce-back Poiseuille profile."""

    NY, NX = 34, 8
    TAU = 0.8
    FORCE = 1e-5
    STEPS = 12000

    @classmethod
    def reference_profile(cls, ny):
        """u(d) = g/(2 nu) d (H - d), d measured from the half-way wall."""
        nu = (cls.TAU - 0.5) / 3.0
        H = ny - 2.0                      # walls at y = 0.5 and y = ny - 1.5
        y = torch.arange(1, ny - 1, dtype=DTYPE)
        d = y - 0.5
        return cls.FORCE / (2 * nu) * d * (H - d)

    def test_profile_below_acceptance_bound(self):
        ny, nx = self.NY, self.NX
        dom = build_sparse_domain(channel_mask(ny, nx), periodic=(True, True))
        solver = SparseAOSLBM(dom, omega=1.0 / self.TAU, dtype=DTYPE)
        solver.initialize(
            torch.ones(ny, nx, dtype=DTYPE),
            torch.zeros(ny, nx, dtype=DTYPE),
            torch.zeros(ny, nx, dtype=DTYPE),
        )
        for _ in range(self.STEPS):
            solver.step(force=(self.FORCE, 0.0))

        _, ux, _ = solver.macroscopic()
        ux_dense = dom.scatter(ux)
        u_num = ux_dense[1:-1, :].mean(dim=1)
        u_ref = self.reference_profile(ny)
        rel_l2 = (
            (u_num - u_ref).pow(2).sum().sqrt() / u_ref.pow(2).sum().sqrt()
        ).item()
        u_max_ratio = (u_num.max() / u_ref.max()).item()

        print(f"\nPoiseuille rel-L2 = {100*rel_l2:.3f}%  u_max/u_ref = {u_max_ratio:.4f}")
        assert rel_l2 < 0.15          # anchor acceptance bound
        assert rel_l2 < 0.03          # documented quality target
        assert 0.97 < u_max_ratio < 1.03