"""Canonical verification cases shipped with TensorDG.

Each module here owns the *definition* of one published test case (initial data,
domain, final time, analytic anchors) so that benchmarks, tests and downstream
callers read the case from a single place instead of copying constants.
"""

from . import elastic_rod_1d
from . import lax_1d
from . import prefinement_exponential
from . import viscous_burgers_1d
from .elastic_rod_1d import (
    CASE_KEY as ELASTIC_ROD_CASE_KEY,
    anchor_table as elastic_rod_anchor_table,
    parameters as elastic_rod_parameters,
    solve_case as solve_elastic_rod_case,
)
from .lax_1d import (
    CASE_KEY as LAX_CASE_KEY,
    DIAPHRAGM as LAX_DIAPHRAGM,
    DOMAIN as LAX_DOMAIN,
    FINAL_TIME as LAX_FINAL_TIME,
    GAMMA as LAX_GAMMA,
    LEFT_STATE as LAX_LEFT_STATE,
    RIGHT_STATE as LAX_RIGHT_STATE,
    lax_parameters,
    solve_lax_case,
    star_state as lax_star_state,
)
from .prefinement_exponential import (
    CASE_KEY as PREFINEMENT_CASE_KEY,
    LEDGER_ANCHOR as PREFINEMENT_LEDGER_ANCHOR,
    REFERENCE as PREFINEMENT_REFERENCE,
    anchor_table as prefinement_anchor_table,
    p_ladder as prefinement_p_ladder,
    prefinement_parameters,
    solve_level as solve_prefinement_level,
    solve_prefinement_case,
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

from .viscous_burgers_1d import CASE_KEY as VISCOUS_BURGERS_CASE_KEY
from .viscous_burgers_1d import solve_viscous_burgers_1d
from .viscous_burgers_1d import viscous_burgers_parameters as parameters

#: alias so the package-level convention (``<case>_parameters``) holds
viscous_burgers_parameters = parameters

__all__ = [
    "CASE_KEY",
    "LAX_CASE_KEY",
    "LAX_DIAPHRAGM",
    "LAX_DOMAIN",
    "LAX_FINAL_TIME",
    "LAX_GAMMA",
    "LAX_LEFT_STATE",
    "LAX_RIGHT_STATE",
    "lax_parameters",
    "solve_lax_case",
    "lax_star_state",
    "VISCOUS_BURGERS_CASE_KEY",
    "viscous_burgers_parameters",
    "solve_viscous_burgers_1d",
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
    "PREFINEMENT_CASE_KEY",
    "PREFINEMENT_LEDGER_ANCHOR",
    "PREFINEMENT_REFERENCE",
    "prefinement_anchor_table",
    "prefinement_p_ladder",
    "prefinement_parameters",
    "solve_prefinement_level",
    "solve_prefinement_case",
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
    "viscous_burgers_1d": VISCOUS_BURGERS_CASE_KEY,
    "lax_1d": LAX_CASE_KEY,
    "prefinement_exponential": PREFINEMENT_CASE_KEY,
}
