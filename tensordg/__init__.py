"""TensorDG: differentiable discontinuous Galerkin methods built with PyTorch."""

from .advection_1d import l2_error, solve_advection
from .advection_2d import error_2d, solve_advection_2d
from .burgers_1d import burgers_error, solve_burgers
from .euler_1d import solve_riemann
from .euler_2d import solve_isentropic_vortex, vortex_error
from .euler_dg_1d import solve_euler_dg_1d
from .heat_1d import heat_error, solve_heat_1d
from .limiters import apply_limiter, apply_tvb_limiter, apply_weno_limiter
from .poisson_2d import poisson_error_2d, solve_poisson_2d

__all__ = [
    "solve_advection", "l2_error",
    "solve_advection_2d", "error_2d",
    "solve_burgers", "burgers_error",
    "solve_heat_1d", "heat_error",
    "solve_poisson_2d", "poisson_error_2d",
    "solve_riemann", "solve_euler_dg_1d",
    "solve_isentropic_vortex", "vortex_error",
    "apply_limiter", "apply_tvb_limiter", "apply_weno_limiter",
]
