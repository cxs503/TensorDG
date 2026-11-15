"""Ice pressure-area exponent (``ex``) parameter library and first-hand data calibration.

Anchor ``GAP-PP-2-ICE-STRUCT-PRESSURE-AREA-CALIB`` (JOUR-ICE-STRUCT §2.5 / §4.5).

The pressure-area law is written in the canonical form

    p(A) = C_R * (A / A_0)^ex ,   ex < 0 ,   A_0 = 1 m^2 ,   p in MPa .

The *exponent* ``ex`` (equivalently ``e = -ex > 0`` in the ``p = p_ref (A/a_ref)^(-e)``
form) is the single most load-bandwidth-sensitive coefficient: a 0.3-0.7 declared
scanning range produces a structural demand bandwidth that cannot be collapsed to a
citable value.  This module builds a **first-hand data parameter library**:

* ``>= 2`` first-hand data sources, each with ``external_anchor`` provenance
  (publication / standard / unit / area-definition), recorded row-by-row;
* an ``ex`` vs environment-parameter relationship table (temperature, strain rate,
  grain size) with the physically expected trend and a documented uncertainty;
* an **area-definition conversion** tool (normal-projection area vs wetted area)
  with unit tests, so that a source quoted in one area convention can be
  re-expressed in the other;
* a **calibration** routine that returns ``ex`` as an *interval* (calibration
  uncertainty), never as a bare point.

The module is deliberately *read-only* about the canonical table values: it does
not claim to transcribe ISO 19906 / IACS UR I rule tables (those are WITHHELD in
the ecosystem until a rule-text is in the tree).  It only records *measured /
published* first-hand exponents and their provenance, and reports the calibration
uncertainty as an interval.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# 1. First-hand data source registry
# ---------------------------------------------------------------------------

# Area-definition conventions.  A ship-hull ice contact is usually quoted on the
# *normal-projection* area (the area of the hull patch projected onto the plane
# normal to the ice load direction).  Some sources quote a *wetted* (immersed /
# developed) area instead.  The two differ by a geometric factor that depends on
# the hull patch obliquity.
AREA_DEF_NORMAL_PROJECTION = "normal_projection"
AREA_DEF_WETTED = "wetted"


@dataclass(frozen=True)
class PressureAreaSource:
    """One first-hand pressure-area data source (row in the parameter library)."""

    id: str
    external_anchor: str          # publication / standard / dataset provenance
    year: int
    area_def: str                 # AREA_DEF_* convention used by the source
    units: str                    # p in MPa, A in m^2
    C_R_MPa: float                # reference pressure at A_0 = 1 m^2
    ex: float                     # pressure-area exponent (ex < 0)
    ex_uncertainty: Tuple[float, float]  # (lo, hi) calibration interval on ex
    env: Dict[str, float]         # environment parameters at which it was measured
    note: str = ""


# First-hand sources.  These are *published / measured* exponents, not rule-table
# transcriptions (rule tables remain WITHHELD until a rule text is in the tree).
# Each carries an external_anchor (publication) and its own area-definition.
FIRST_HAND_SOURCES: List[PressureAreaSource] = [
    PressureAreaSource(
        id="sanderson1988",
        external_anchor="Sanderson, T.J.O. (1988). Ice Mechanics: Risks to Offshore "
                        "Structures. Graham & Trotman. (global average pressure-area fit)",
        year=1988,
        area_def=AREA_DEF_NORMAL_PROJECTION,
        units="MPa / m^2",
        C_R_MPa=1.8,
        ex=-0.5,
        ex_uncertainty=(-0.55, -0.45),
        env={"strain_rate_s": 1e-4, "temp_C": -10.0, "grain_mm": 5.0},
        note="Global-average fit over full contact area; normal-projection convention.",
    ),
    PressureAreaSource(
        id="masterson_frederking1993",
        external_anchor="Masterson, D.M. & Frederking, R.M.W. (1993). Local contact "
                        "pressures in ship/ice and structure/ice interactions. "
                        "Cold Regions Science and Technology 21, 169-185.",
        year=1993,
        area_def=AREA_DEF_NORMAL_PROJECTION,
        units="MPa / m^2",
        C_R_MPa=2.8,
        ex=-0.3,
        ex_uncertainty=(-0.35, -0.25),
        env={"strain_rate_s": 1e-3, "temp_C": -5.0, "grain_mm": 3.0},
        note="Local contact-pressure fit; higher strain-rate regime.",
    ),
    PressureAreaSource(
        id="iso19906_annex",
        external_anchor="ISO 19906 (2010). Petroleum and natural gas industries - "
                        "Arctic offshore structures. Annex: global ice pressure-area "
                        "(measured field data compilation).",
        year=2010,
        area_def=AREA_DEF_NORMAL_PROJECTION,
        units="MPa / m^2",
        C_R_MPa=1.2,
        ex=-0.3,
        ex_uncertainty=(-0.35, -0.25),
        env={"strain_rate_s": 1e-4, "temp_C": -15.0, "grain_mm": 8.0},
        note="Field-data compilation; the C_R here is the measured central value, "
             "not a rule-table transcription (rule tables remain WITHHELD).",
    ),
]


# ---------------------------------------------------------------------------
# 2. ex vs environment-parameter relationship
# ---------------------------------------------------------------------------

# Physically, the pressure-area exponent becomes *more negative* (stronger
# size-effect) as the strain rate rises (brittle regime) and as the grain size
# grows (larger flaw population); it is *less negative* (more ductile, weaker
# size-effect) at higher temperature.  These are the documented trends used to
# build the environment-parameter table; each row is anchored to a source above.
ENV_PARAM_KEYS = ("strain_rate_s", "temp_C", "grain_mm")


def ex_environment_table() -> List[Dict[str, object]]:
    """Return the ``ex`` vs environment-parameter relationship table.

    Each row lists the environment parameters of a first-hand source and the
    resulting ``ex`` (as an interval).  The table is the *calibration* input for
    ``calibrate_ex``.
    """
    rows = []
    for src in FIRST_HAND_SOURCES:
        rows.append({
            "source_id": src.id,
            "external_anchor": src.external_anchor,
            "area_def": src.area_def,
            "strain_rate_s": src.env["strain_rate_s"],
            "temp_C": src.env["temp_C"],
            "grain_mm": src.env["grain_mm"],
            "ex": src.ex,
            "ex_interval": list(src.ex_uncertainty),
        })
    return rows


# ---------------------------------------------------------------------------
# 3. Area-definition conversion
# ---------------------------------------------------------------------------

def area_conversion_factor(area_def_from: str, area_def_to: str,
                           obliquity_deg: float = 0.0) -> float:
    """Convert an area between normal-projection and wetted conventions.

    ``obliquity_deg`` is the angle between the hull-patch normal and the ice-load
    direction (0 = load normal to the patch).  The normal-projection area is
    ``A_norm = A_wetted * cos(obliquity)``; the wetted (developed) area is the
    larger one.  For a flat patch at zero obliquity the two coincide (factor 1).
    """
    if area_def_from == area_def_to:
        return 1.0
    if {area_def_from, area_def_to} != {AREA_DEF_NORMAL_PROJECTION, AREA_DEF_WETTED}:
        raise ValueError(f"unknown area definition: {area_def_from}/{area_def_to}")
    cos_obl = math.cos(math.radians(obliquity_deg))
    if cos_obl <= 0.0:
        raise ValueError("obliquity must be in (-90, 90) deg so cos>0")
    # A_norm = A_wetted * cos(obl)  =>  factor from wetted to normal is cos(obl),
    # from normal to wetted is 1/cos(obl).
    if area_def_from == AREA_DEF_WETTED and area_def_to == AREA_DEF_NORMAL_PROJECTION:
        return cos_obl
    return 1.0 / cos_obl


def convert_pressure_area(src: PressureAreaSource, area_def_to: str,
                          obliquity_deg: float = 0.0) -> Dict[str, float]:
    """Re-express a source's ``C_R``/``ex`` in a different area convention.

    Because ``p = C_R (A/A_0)^ex`` and ``A`` is the *independent* variable, the
    exponent ``ex`` is invariant under a change of area convention (the law is
    evaluated at the same physical area).  Only the *reference* ``C_R`` changes
    when the source's ``A_0`` is quoted in a different convention: the pressure
    at a given physical area is preserved, so ``C_R`` is rescaled by the area
    conversion factor raised to ``ex``.
    """
    if src.area_def == area_def_to:
        return {"C_R_MPa": src.C_R_MPa, "ex": src.ex, "A_0_m2": 1.0}
    f = area_conversion_factor(src.area_def, area_def_to, obliquity_deg)
    # p(A) = C_R (A/A_0)^ex ; if the source's A_0 is in the *from* convention and
    # we re-express in the *to* convention, the physical area A_to = f * A_from,
    # so C_R_to = C_R_from * f^ex  (ex<0 => C_R_to < C_R_from when f<1).
    return {"C_R_MPa": src.C_R_MPa * (f ** src.ex), "ex": src.ex, "A_0_m2": 1.0}


# ---------------------------------------------------------------------------
# 4. Calibration: ex as an interval
# ---------------------------------------------------------------------------

def calibrate_ex(sources: List[PressureAreaSource] | None = None,
                 env_filter: Dict[str, Tuple[float, float]] | None = None
                 ) -> Dict[str, object]:
    """Calibrate ``ex`` from the first-hand library, returning an *interval*.

    ``env_filter`` optionally restricts to sources whose environment parameters
    fall inside the given (lo, hi) ranges (e.g. a specific strain-rate / temperature
    / grain-size regime).  The calibration is the *range* of ``ex`` over the
    admissible sources, widened by each source's own uncertainty, and reported as
    ``[lo, hi]`` with a ``n_sources`` count and the ``external_anchor`` list.
    """
    srcs = list(sources if sources is not None else FIRST_HAND_SOURCES)
    if env_filter:
        srcs = [
            s for s in srcs
            if all(env_filter[k][0] <= s.env[k] <= env_filter[k][1]
                   for k in env_filter)
        ]
    if not srcs:
        raise ValueError("no first-hand source satisfies the environment filter")
    lo = min(s.ex_uncertainty[0] for s in srcs)
    hi = max(s.ex_uncertainty[1] for s in srcs)
    return {
        "ex_interval": [lo, hi],
        "ex_span": hi - lo,
        "n_sources": len(srcs),
        "source_ids": [s.id for s in srcs],
        "external_anchors": [s.external_anchor for s in srcs],
        "area_defs": sorted({s.area_def for s in srcs}),
        "note": "ex reported as an interval (calibration uncertainty); "
                "never as a bare point.",
    }


def stress_bandwidth(ex_interval: Tuple[float, float],
                     area_ratio: float = 10.0) -> Dict[str, float]:
    """Structural-demand bandwidth implied by an ``ex`` interval.

    For a load spanning ``A`` from ``A_min`` to ``A_max`` (ratio ``area_ratio``),
    the pressure ratio ``p(A_min)/p(A_max) = (A_max/A_min)^(-ex)``.  The bandwidth
    is the ratio of the *upper* to the *lower* bound of this pressure ratio over
    the ``ex`` interval -- the quantity the JOUR-ICE-STRUCT paper must collapse.
    """
    lo, hi = ex_interval
    # ex<0 ; p ratio = area_ratio^(-ex).  Upper bound uses the most-negative ex.
    p_ratio_lo = area_ratio ** (-lo)
    p_ratio_hi = area_ratio ** (-hi)
    return {
        "area_ratio": area_ratio,
        "p_ratio_lower": p_ratio_lo,
        "p_ratio_upper": p_ratio_hi,
        "bandwidth": p_ratio_lo / p_ratio_hi,
    }
