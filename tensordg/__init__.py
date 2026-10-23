"""TensorDG: differentiable discontinuous Galerkin methods built with PyTorch."""

from .advection_1d import solve_advection, l2_error

__all__ = ["solve_advection", "l2_error"]
