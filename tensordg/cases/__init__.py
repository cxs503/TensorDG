"""Canonical verification cases shipped with TensorDG.

Each module here owns the *definition* of one published test case (initial data,
domain, final time, analytic anchors) so that benchmarks, tests and downstream
callers read the case from a single place instead of copying constants.
"""

from .shock_entropy import (
    CASE_KEY,
    DIAPHRAGM,
    DOMAIN,
    FINAL_TIME,
    GAMMA,
    LEFT_STATE,
    exact_mass_budget,
    initial_state,
    rankine_hugoniot_anchor,
    solve_dg_case,
)

__all__ = [
    "CASE_KEY",
    "DIAPHRAGM",
    "DOMAIN",
    "FINAL_TIME",
    "GAMMA",
    "LEFT_STATE",
    "exact_mass_budget",
    "initial_state",
    "rankine_hugoniot_anchor",
    "solve_dg_case",
]