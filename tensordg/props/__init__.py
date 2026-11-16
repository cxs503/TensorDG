"""TensorDG physical property database (v1).

A provenance-carrying property store for solver material/fluid data:
water (fresh / sea), air, ice (sea / fresh), sediment and structural
materials.  SI base units are authoritative; every value carries a citation
and an explicit applicability domain.

Typical use from a solver::

    from tensordg.props import load_db
    db = load_db()
    nu = db.get("sea_water", "nu", T=15.0)          # m^2/s, ITTC table
    mu = db.get("air", "mu", T=20.0)                # Pa*s, Sutherland
    E  = db.get("AH36", "E", T=20.0)                # Pa, rule value

Command line::

    python -m tensordg.props list
    python -m tensordg.props query water nu --T 15 --unit cSt
    python -m tensordg.props validate
"""

from __future__ import annotations

from .db import (  # noqa: F401
    DATA_DIR,
    MODELS,
    PROPERTY_ALIASES,
    OutOfDomain,
    PropsDB,
    PropsError,
    QueryResult,
    UnknownProperty,
    UnknownSubstance,
    load_db,
)
from .schema import Property, Substance, validate_payload  # noqa: F401
from .units import SI_BASE, UNIT_TABLE, UnitError, dimension_of, from_si, to_si  # noqa: F401

__all__ = [
    "DATA_DIR", "MODELS", "PROPERTY_ALIASES", "OutOfDomain", "PropsDB",
    "PropsError", "QueryResult", "UnknownProperty", "UnknownSubstance",
    "load_db", "Property", "Substance", "validate_payload", "SI_BASE",
    "UNIT_TABLE", "UnitError", "dimension_of", "from_si", "to_si",
]
__version__ = "1.0.0"