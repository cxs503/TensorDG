"""Backward-compatible Sod wrappers around the general exact Riemann solver."""
from __future__ import annotations

from benchmarks.riemann_exact import GAMMA, cell_averages as _cell_averages
from benchmarks.riemann_exact import sample as _sample, star_state as _star_state

LEFT = (1.0, 0.0, 1.0)
RIGHT = (0.125, 0.0, 0.1)


def star_state():
    return _star_state(LEFT, RIGHT, GAMMA)


def sample(x, final_time: float = 0.2, diaphragm: float = 0.5):
    return _sample(x, final_time, LEFT, RIGHT, diaphragm, GAMMA)


def cell_averages(ncells: int, final_time: float = 0.2, quadrature_order: int = 8):
    return _cell_averages(
        ncells, final_time, LEFT, RIGHT, diaphragm=0.5,
        domain=(0.0, 1.0), quadrature_order=quadrature_order, gamma=GAMMA,
    )
