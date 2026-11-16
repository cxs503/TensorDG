"""Unit registry for the TensorDG physical property database.

Every stored value carries an explicit unit string.  The registry maps that
string onto ``(factor_to_SI, dimension)`` so that queries can be returned in
SI base units or in any registered engineering unit without ambiguity.

Only SI base units are authoritative (acceptance item (3) of the property
database feature: "unit convention, SI base").  Derived units are pure
conversion factors, nothing else.
"""

from __future__ import annotations

from typing import Dict, Tuple

#: dimension -> SI base unit used internally
SI_BASE: Dict[str, str] = {
    "density": "kg/m^3",
    "viscosity_kinematic": "m^2/s",
    "viscosity_dynamic": "Pa*s",
    "stress": "Pa",
    "modulus": "Pa",
    "pressure": "Pa",
    "temperature": "K",
    "length": "m",
    "velocity": "m/s",
    "angle": "rad",
    "dimensionless": "1",
    "expansion": "1/K",
    "density_bulk": "kg/m^3",
    "specific_heat": "J/(kg*K)",
    "conductivity": "W/(m*K)",
}

#: unit string -> (factor to SI base of its dimension, dimension)
UNIT_TABLE: Dict[str, Tuple[float, str]] = {
    # --- density -----------------------------------------------------------
    "kg/m^3": (1.0, "density"),
    "kg/m3": (1.0, "density"),
    "kg.m^-3": (1.0, "density"),
    "g/cm^3": (1.0e3, "density"),
    "g/cm3": (1.0e3, "density"),
    "kg/L": (1.0e3, "density"),
    "t/m^3": (1.0e3, "density"),
    # --- kinematic viscosity ----------------------------------------------
    "m^2/s": (1.0, "viscosity_kinematic"),
    "m2/s": (1.0, "viscosity_kinematic"),
    "m^2.s^-1": (1.0, "viscosity_kinematic"),
    "mm^2/s": (1.0e-6, "viscosity_kinematic"),
    "cSt": (1.0e-6, "viscosity_kinematic"),
    # --- dynamic viscosity ------------------------------------------------
    "Pa*s": (1.0, "viscosity_dynamic"),
    "Pa.s": (1.0, "viscosity_dynamic"),
    "Pa s": (1.0, "viscosity_dynamic"),
    "kg/(m*s)": (1.0, "viscosity_dynamic"),
    "mPa*s": (1.0e-3, "viscosity_dynamic"),
    "mPa.s": (1.0e-3, "viscosity_dynamic"),
    "cP": (1.0e-3, "viscosity_dynamic"),
    "P": (1.0e-1, "viscosity_dynamic"),
    # --- stress / modulus / pressure --------------------------------------
    "Pa": (1.0, "stress"),
    "kPa": (1.0e3, "stress"),
    "MPa": (1.0e6, "stress"),
    "GPa": (1.0e9, "stress"),
    "N/m^2": (1.0, "stress"),
    "N/mm^2": (1.0e6, "stress"),
    # --- misc -------------------------------------------------------------
    "1": (1.0, "dimensionless"),
    "-": (1.0, "dimensionless"),
    "1/K": (1.0, "expansion"),
    "1/degC": (1.0, "expansion"),
    "m": (1.0, "length"),
    "cm": (1.0e-2, "length"),
    "mm": (1.0e-3, "length"),
    "m/s": (1.0, "velocity"),
    "J/(kg*K)": (1.0, "specific_heat"),
    "W/(m*K)": (1.0, "conductivity"),
    "deg": (3.141592653589793 / 180.0, "angle"),
    "rad": (1.0, "angle"),
}

#: temperature axis units -> callable(value) -> degC
_T_TO_DEGC = {
    "degC": lambda v: float(v),
    "C": lambda v: float(v),
    "°C": lambda v: float(v),
    "K": lambda v: float(v) - 273.15,
    "degF": lambda v: (float(v) - 32.0) * 5.0 / 9.0,
    "F": lambda v: (float(v) - 32.0) * 5.0 / 9.0,
}


class UnitError(ValueError):
    """Unknown unit string or dimension mismatch."""


def to_si(value: float, unit: str) -> float:
    """Convert ``value`` expressed in ``unit`` to SI base of its dimension."""
    key = _lookup(unit)
    return float(value) * UNIT_TABLE[key][0]


def from_si(value_si: float, unit: str) -> float:
    """Convert an SI base value into ``unit``."""
    key = _lookup(unit)
    return float(value_si) / UNIT_TABLE[key][0]


def dimension_of(unit: str) -> str:
    return UNIT_TABLE[_lookup(unit)][1]


def temperature_to_degc(value: float, unit: str = "degC") -> float:
    """Convert a temperature (axis coordinate) to degrees Celsius."""
    key = unit.strip()
    if key not in _T_TO_DEGC:
        raise UnitError(f"unknown temperature unit {unit!r}; "
                        f"known: {sorted(_T_TO_DEGC)}")
    return _T_TO_DEGC[key](value)


def _lookup(unit: str) -> str:
    key = unit.strip()
    if key not in UNIT_TABLE:
        # tolerate a few cosmetic spellings
        alt = key.replace("**", "^").replace("·", "*")
        if alt in UNIT_TABLE:
            return alt
        raise UnitError(f"unknown unit {unit!r}; known units: "
                        f"{sorted(UNIT_TABLE)}")
    return key