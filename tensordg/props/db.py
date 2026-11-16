"""Query engine for the TensorDG physical property database.

Public surface (acceptance item (3): "query API + CLI + unit convention")::

    from tensordg.props import load_db
    db = load_db()
    res = db.query("water", "nu", T=15.0, unit="cSt")
    res.value          # 1.1386  (cSt)
    res.value_si       # 1.1386e-06 (m^2/s)
    res.source["ref"]  # provenance string, always present

Design rules
------------
* SI base units are authoritative (:mod:`tensordg.props.units`).
* Every returned value is accompanied by its provenance and the domain it was
  evaluated in -- no value can be fetched without a citation.
* Evaluations outside the declared applicability domain raise
  :class:`OutOfDomain` unless the caller explicitly asks for clamping.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import units
from .schema import Property, Substance, validate_payload

DATA_DIR = Path(__file__).with_name("data")
_DATA_FILES = ("water.json", "air.json", "ice.json", "sediment.json",
               "structural.json")

#: short aliases so solver code can keep writing rho / nu / mu / E
PROPERTY_ALIASES: Dict[str, str] = {
    "rho": "density",
    "rho_water": "density",
    "nu": "viscosity_kinematic",
    "mu": "viscosity_dynamic",
    "e": "youngs_modulus",
    "E": "youngs_modulus",
    "sigma_f": "flexural_strength",
    "sigma_c": "compressive_strength",
    "fy": "yield_strength",
    "fu": "ultimate_strength",
    "c": "speed_of_sound",
    "cp": "specific_heat_capacity",
    "k": "thermal_conductivity",
}


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
class PropsError(Exception):
    """Base class for property database errors."""


class UnknownSubstance(PropsError, KeyError):
    pass


class UnknownProperty(PropsError, KeyError):
    pass


class OutOfDomain(PropsError, ValueError):
    """Requested condition lies outside the declared applicability domain."""


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def _vogel_water(T_degc: float, params: Dict[str, float], P: float) -> float:
    """Vogel-type correlation: mu = A * 10**(B / (T[K] - C))."""
    T = T_degc + 273.15
    denom = T - params["C"]
    if denom <= 0.0:
        raise OutOfDomain("vogel correlation undefined for T <= C")
    return params["A"] * 10.0 ** (params["B"] / denom)


def _sutherland_gas(T_degc: float, params: Dict[str, float], P: float) -> float:
    """Sutherland law: mu = mu_ref (T/T_ref)^1.5 (T_ref+S)/(T+S)."""
    T = T_degc + 273.15
    T_ref = params["T_ref"]
    S = params["S"]
    return params["mu_ref"] * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)


def _ideal_gas_density(T_degc: float, params: Dict[str, float], P: float) -> float:
    """Ideal-gas density rho = p / (R T)."""
    T = T_degc + 273.15
    return P / (params["R"] * T)


def _ideal_gas_sound_speed(T_degc: float, params: Dict[str, float],
                           P: float) -> float:
    """Sound speed a = sqrt(gamma R T)."""
    T = T_degc + 273.15
    return math.sqrt(params["gamma"] * params["R"] * T)


MODELS = {
    "vogel_water": _vogel_water,
    "sutherland_gas": _sutherland_gas,
    "ideal_gas_density": _ideal_gas_density,
    "ideal_gas_sound_speed": _ideal_gas_sound_speed,
}


# --------------------------------------------------------------------------- #
# result object
# --------------------------------------------------------------------------- #
@dataclass
class QueryResult:
    substance: str
    substance_class: str
    prop: str
    value_si: float
    unit: str
    value: float
    conditions: Dict[str, Any]
    source: Dict[str, Any] = field(default_factory=dict)
    domain: Dict[str, List[float]] = field(default_factory=dict)
    method: str = ""
    accuracy: str = ""
    applicability: str = ""
    clamped: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "substance": self.substance,
            "class": self.substance_class,
            "property": self.prop,
            "value": self.value,
            "unit": self.unit,
            "value_si": self.value_si,
            "si_unit": units.SI_BASE.get(
                units.dimension_of(self.unit), "?"),
            "conditions": self.conditions,
            "provenance": self.source,
            "domain": self.domain,
            "method": self.method,
            "accuracy": self.accuracy,
            "applicability": self.applicability,
            "clamped": self.clamped,
        }


# --------------------------------------------------------------------------- #
# database
# --------------------------------------------------------------------------- #
class PropsDB:
    """In-memory physical property database with provenance-preserving queries."""

    def __init__(self, substances: Iterable[Substance], meta: Optional[Dict] = None):
        self._subs: Dict[str, Substance] = {}
        self._alias: Dict[str, str] = {}
        for sub in substances:
            self._subs[sub.name] = sub
            self._alias[sub.name.lower()] = sub.name
            self._alias[sub.name.lower().replace("_", " ")] = sub.name
            for alias in sub.aliases:
                self._alias[alias.lower()] = sub.name
        self.meta = dict(meta or {})

    # -- construction ----------------------------------------------------- #
    @classmethod
    def from_payloads(cls, payloads: List[Dict[str, Any]],
                      meta: Optional[Dict] = None) -> "PropsDB":
        substances = [Substance.from_dict(s) for s in payloads]
        db = cls(substances, meta)
        for sub in substances:  # resolve class-name aliases when unambiguous
            owners = [s.name for s in substances if s.klass == sub.klass]
            if len(owners) == 1:
                db._alias.setdefault(sub.klass.lower(), sub.name)
        return db

    @classmethod
    def load(cls, data_dir: Optional[Path] = None,
             files: Optional[Iterable[str]] = None) -> "PropsDB":
        data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
        payloads: List[Dict[str, Any]] = []
        meta: Dict[str, Any] = {"files": []}
        for name in (files if files is not None else _DATA_FILES):
            path = data_dir / name
            if not path.exists():
                raise PropsError(f"missing data file {path}")
            blob = json.loads(path.read_text())
            meta["files"].append(str(name))
            meta.setdefault("schema_version", blob.get("schema_version"))
            payloads.extend(blob["substances"])
        db = cls.from_payloads(payloads, meta)
        db.raw_payloads = payloads  # kept for validate()
        return db

    # -- introspection ---------------------------------------------------- #
    def substances(self) -> List[str]:
        return sorted(self._subs)

    def classes(self) -> List[str]:
        return sorted({s.klass for s in self._subs.values()})

    def properties(self, substance: str) -> List[str]:
        return sorted(self._resolve_substance(substance).properties)

    def row_count(self) -> int:
        return sum(s.n_rows for s in self._subs.values())

    def summary(self) -> List[Dict[str, Any]]:
        return [
            {
                "substance": sub.name,
                "class": sub.klass,
                "state": sub.state,
                "properties": sorted(sub.properties),
                "rows": sub.n_rows,
            }
            for sub in sorted(self._subs.values(), key=lambda s: (s.klass, s.name))
        ]

    def provenance_index(self) -> List[Dict[str, Any]]:
        """Flat list of (substance, property, source) triples."""
        out = []
        for sub in sorted(self._subs.values(), key=lambda s: s.name):
            for name, prop in sorted(sub.properties.items()):
                out.append({"substance": sub.name, "property": name,
                            "rows": prop.n_rows, "source": dict(prop.source),
                            "domain": prop.domain, "unit": prop.unit,
                            "accuracy": prop.accuracy})
        return out

    def validate(self) -> List[str]:
        payloads = getattr(self, "raw_payloads", None)
        if payloads is None:
            payloads = [
                {"name": s.name, "class": s.klass, "aliases": s.aliases,
                 "state": s.state,
                 "properties": {k: _prop_to_payload(v)
                                for k, v in s.properties.items()}}
                for s in self._subs.values()
            ]
        problems = validate_payload(payloads)
        # cross-record consistency: derived properties must resolve
        for sub in self._subs.values():
            for name, prop in sub.properties.items():
                if prop.mode == "derived":
                    for token in _expr_tokens(prop.expr):
                        if token not in sub.properties:
                            problems.append(
                                f"{sub.name}/{name}: derived expr references "
                                f"unknown property {token!r}")
                if prop.mode == "correlation" and prop.model not in MODELS:
                    problems.append(
                        f"{sub.name}/{name}: unknown correlation model "
                        f"{prop.model!r}")
        return problems

    # -- query ------------------------------------------------------------ #
    def query(self, substance: str, prop: str, T: Optional[float] = None,
              P: float = 101325.0, T_unit: str = "degC",
              unit: Optional[str] = None, clamp: bool = False,
              T_K: Optional[float] = None) -> QueryResult:
        """Evaluate one property.

        Parameters
        ----------
        substance : substance name or alias (``"water"``, ``"sea_water"``, ...)
        prop      : property name or alias (``"nu"``, ``"rho"``, ``"E"``, ...)
        T         : temperature on the property's axis (degC by default)
        P         : absolute pressure in Pa (used by gas-law models)
        T_unit    : unit of ``T`` (``"degC"``, ``"K"``, ``"degF"``)
        unit      : output unit; defaults to the unit the record is stored in
        clamp     : return the domain-boundary value instead of raising

        Returns
        -------
        QueryResult with ``value_si`` (SI base), ``value`` (requested unit),
        provenance dict, and ``clamped`` flag.
        """
        sub = self._resolve_substance(substance)
        prop_rec = self._resolve_property(sub, prop)
        if T is None and T_K is not None:
            T, T_unit = T_K, "K"
        if T is None:
            raise PropsError(
                f"{sub.name}/{prop_rec.name}: a temperature is required "
                f"(pass T=... or T_K=...)"
            )
        T_degc = units.temperature_to_degc(T, T_unit)
        T_degc_eval, clamped = self._apply_domain(prop_rec, T_degc, P, clamp)

        value_store = self._evaluate(sub, prop_rec, T_degc_eval, P)
        value_si = units.to_si(value_store, prop_rec.unit)
        out_unit = unit if unit is not None else prop_rec.unit
        dim_store = units.dimension_of(prop_rec.unit)
        if units.dimension_of(out_unit) != dim_store:
            raise units.UnitError(
                f"cannot express {prop_rec.name} ({dim_store}) in {out_unit!r}")
        value = units.from_si(value_si, out_unit)
        return QueryResult(
            substance=sub.name,
            substance_class=sub.klass,
            prop=prop_rec.name,
            value_si=value_si,
            unit=out_unit,
            value=value,
            conditions={"T_degC": T_degc, "T_used_degC": T_degc_eval,
                        "P_Pa": P, "requested_T_unit": T_unit},
            source=dict(prop_rec.source),
            domain=prop_rec.domain,
            method=prop_rec.mode + (f":{prop_rec.model}" if prop_rec.model else ""),
            accuracy=prop_rec.accuracy,
            applicability=prop_rec.applicability,
            clamped=clamped,
        )

    def get(self, substance: str, prop: str, **kwargs) -> float:
        """Convenience wrapper returning only the SI value."""
        return self.query(substance, prop, **kwargs).value_si

    # -- internals -------------------------------------------------------- #
    def _resolve_substance(self, substance: str) -> Substance:
        key = str(substance).strip().lower()
        if key in self._alias:
            return self._subs[self._alias[key]]
        if key in self._subs:
            return self._subs[key]
        raise UnknownSubstance(
            f"unknown substance {substance!r}; known: {self.substances()}")

    def _resolve_property(self, sub: Substance, prop: str) -> Property:
        name = str(prop).strip()
        if name in sub.properties:
            return sub.properties[name]
        alias = PROPERTY_ALIASES.get(name) or PROPERTY_ALIASES.get(name.lower())
        if alias and alias in sub.properties:
            return sub.properties[alias]
        cand = name.lower()
        for key in sub.properties:
            if key.lower() == cand or key.replace("_", "") == cand.replace("_", ""):
                return sub.properties[key]
        raise UnknownProperty(
            f"substance {sub.name!r} has no property {prop!r}; "
            f"available: {sorted(sub.properties)}")

    @staticmethod
    def _apply_domain(prop: Property, T_degc: float, P: float,
                      clamp: bool) -> tuple:
        """Check (T, P) against the declared domain; optionally clamp."""
        clamped = False
        T_used = T_degc
        T_lo, T_hi = prop.domain.get(prop.x.get("name", "T"), [None, None])
        if T_lo is not None and not (T_lo - 1e-9 <= T_degc <= T_hi + 1e-9):
            if not clamp:
                raise OutOfDomain(
                    f"{prop.name}: T={T_degc:.3f} degC outside domain "
                    f"[{T_lo}, {T_hi}] degC "
                    f"(source: {prop.source.get('ref')})")
            T_used = min(max(T_degc, T_lo), T_hi)
            clamped = True
        P_lo, P_hi = prop.domain.get("P", [None, None])
        if P_lo is not None and not (P_lo - 1e-6 <= P <= P_hi + 1e-6):
            if not clamp:
                raise OutOfDomain(
                    f"{prop.name}: P={P:.1f} Pa outside domain "
                    f"[{P_lo}, {P_hi}] Pa "
                    f"(source: {prop.source.get('ref')})")
            clamped = True
        return T_used, clamped

    def _evaluate(self, sub: Substance, prop: Property, T_degc: float,
                  P: float, _depth: int = 0) -> float:
        if _depth > 4:
            raise PropsError(f"{sub.name}/{prop.name}: derived chain too deep")
        if prop.mode == "table":
            return _interp(prop.points, T_degc, prop)
        if prop.mode == "constant":
            return float(prop.value)
        if prop.mode == "correlation":
            model = MODELS.get(prop.model)
            if model is None:
                raise PropsError(
                    f"{sub.name}/{prop.name}: unknown model {prop.model!r}")
            return float(model(T_degc, prop.params, P))
        if prop.mode == "derived":
            tokens = _expr_tokens(prop.expr)
            values = {}
            for token in tokens:
                dep = self._resolve_property(sub, token)
                dep_T, _ = self._apply_domain(dep, T_degc, P, clamp=True)
                values[token] = self._evaluate(sub, dep, dep_T, P, _depth + 1)
            expr = prop.expr
            for token in sorted(values, key=len, reverse=True):
                expr = expr.replace(token, repr(values[token]))
            return float(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307
        raise PropsError(f"{sub.name}/{prop.name}: unsupported mode {prop.mode!r}")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _interp(points: List[List[float]], x: float, prop: Property) -> float:
    """Piecewise-linear interpolation over an increasing abscissa."""
    if prop.interp not in ("linear", "piecewise_linear"):
        raise PropsError(f"{prop.name}: unsupported interpolation {prop.interp!r}")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    x0, x1, y0, y1 = xs[lo], xs[hi], ys[lo], ys[hi]
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _expr_tokens(expr: str) -> List[str]:
    tokens, cur = [], ""
    for ch in expr:
        if ch.isalnum() or ch == "_":
            cur += ch
        else:
            if cur:
                tokens.append(cur)
            cur = ""
    if cur:
        tokens.append(cur)
    return tokens


def _prop_to_payload(prop: Property) -> Dict[str, Any]:
    payload = {
        "dimension": prop.dimension, "unit": prop.unit, "mode": prop.mode,
        "source": prop.source, "domain": prop.domain, "x": prop.x,
        "points": prop.points, "interp": prop.interp,
        "accuracy": prop.accuracy, "applicability": prop.applicability,
    }
    if prop.model:
        payload["model"] = prop.model
    if prop.params:
        payload["params"] = prop.params
    if prop.expr:
        payload["expr"] = prop.expr
    if prop.value is not None:
        payload["value"] = prop.value
    return payload


def load_db(data_dir: Optional[Path] = None,
            files: Optional[Iterable[str]] = None) -> PropsDB:
    """Load the shipped property database."""
    return PropsDB.load(data_dir=data_dir, files=files)