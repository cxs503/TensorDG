"""Canonical verification cases shipped with TensorDG.

Each module here owns the *definition* of one published test case (initial data,
domain, final time, analytic anchors) so that benchmarks, tests and downstream
callers read the case from a single place instead of copying constants.
"""

from . import elastic_rod_1d
from .elastic_rod_1d import (
    CASE_KEY as ELASTIC_ROD_CASE_KEY,
    anchor_table as elastic_rod_anchor_table,
    parameters as elastic_rod_parameters,
    solve_case as solve_elastic_rod_case,
)
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
    "elastic_rod_1d",
    "ELASTIC_ROD_CASE_KEY",
    "elastic_rod_anchor_table",
    "elastic_rod_parameters",
    "solve_elastic_rod_case",
    "CASES",
]

#: case key -> canonical case identifier for every case module in this package
CASES = {
    "shock_entropy": CASE_KEY,
    "elastic_rod_1d": ELASTIC_ROD_CASE_KEY,
}