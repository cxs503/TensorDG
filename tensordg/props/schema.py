"""Record schema and validation for the TensorDG physical property database.

Schema (feature acceptance item (1))::

    substance  x  condition (T, P)  x  property  x  provenance  x  domain

Every property record must carry a non-empty provenance block
``{"ref": <citation>, "type": table|correlation|derived|constant}`` plus an
explicit applicability domain.  :func:`validate_payload` enforces this and is
run by the test-suite over the whole shipped database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .units import UnitError, dimension_of

VALID_MODES = {"table", "constant", "correlation", "derived"}
VALID_SOURCE_TYPES = {"table", "correlation", "derived", "constant"}
_REQUIRED_PROPERTY_KEYS = ("dimension", "unit", "mode", "source", "domain")


@dataclass
class Property:
    """A single physical property of a single substance."""

    name: str
    dimension: str
    unit: str
    mode: str
    domain: Dict[str, List[float]]
    source: Dict[str, Any]
    x: Dict[str, str] = field(default_factory=lambda: {"name": "T", "unit": "degC"})
    points: List[List[float]] = field(default_factory=list)
    value: Any = None
    model: str = ""
    params: Dict[str, float] = field(default_factory=dict)
    expr: str = ""
    interp: str = "linear"
    accuracy: str = ""
    applicability: str = ""
    extra_inputs: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, name: str, payload: Dict[str, Any]) -> "Property":
        missing = [k for k in _REQUIRED_PROPERTY_KEYS if k not in payload]
        if missing:
            raise ValueError(f"property {name!r}: missing keys {missing}")
        points = [[float(a), float(b)] for a, b in payload.get("points", [])]
        return cls(
            name=name,
            dimension=payload["dimension"],
            unit=payload["unit"],
            mode=payload["mode"],
            domain={k: [float(v[0]), float(v[1])]
                    for k, v in payload["domain"].items()},
            source=dict(payload["source"]),
            x=dict(payload.get("x", {"name": "T", "unit": "degC"})),
            points=points,
            value=payload.get("value"),
            model=payload.get("model", ""),
            params=dict(payload.get("params", {})),
            expr=payload.get("expr", ""),
            interp=payload.get("interp", "linear"),
            accuracy=payload.get("accuracy", ""),
            applicability=payload.get("applicability", ""),
            extra_inputs=list(payload.get("extra_inputs", [])),
        )

    # ------------------------------------------------------------------ #
    @property
    def n_rows(self) -> int:
        """Number of stored data rows contributed by this record."""
        if self.mode == "table":
            return len(self.points)
        return 1

    def check(self) -> List[str]:
        """Return a list of schema violations (empty means valid)."""
        problems: List[str] = []
        if self.mode not in VALID_MODES:
            problems.append(f"{self.name}: invalid mode {self.mode!r}")
        if self.source.get("type") not in VALID_SOURCE_TYPES:
            problems.append(
                f"{self.name}: invalid source.type {self.source.get('type')!r}")
        if not str(self.source.get("ref", "")).strip():
            problems.append(f"{self.name}: empty provenance source.ref")
        try:
            dim = dimension_of(self.unit)
        except UnitError as exc:  # pragma: no cover - defensive
            problems.append(f"{self.name}: {exc}")
            dim = None
        if dim is not None and dim != self.dimension:
            problems.append(
                f"{self.name}: unit {self.unit!r} has dimension {dim!r} "
                f"but record declares {self.dimension!r}")
        if self.mode == "table":
            if len(self.points) < 2:
                problems.append(f"{self.name}: table needs >= 2 points")
            xs = [p[0] for p in self.points]
            if any(b <= a for a, b in zip(xs, xs[1:])):
                problems.append(
                    f"{self.name}: table abscissa not strictly increasing")
            lo, hi = self.domain.get(self.x["name"], [None, None])
            if lo is not None and (xs[0] < lo - 1e-9 or xs[-1] > hi + 1e-9):
                problems.append(
                    f"{self.name}: table points [{xs[0]}, {xs[-1]}] outside "
                    f"declared domain [{lo}, {hi}]")
        if self.mode == "correlation" and not self.model:
            problems.append(f"{self.name}: correlation without model name")
        if self.mode == "derived" and not self.expr:
            problems.append(f"{self.name}: derived property without expr")
        if self.mode == "constant" and self.value is None:
            problems.append(f"{self.name}: constant property without value")
        return problems


@dataclass
class Substance:
    name: str
    klass: str
    properties: Dict[str, Property]
    aliases: List[str] = field(default_factory=list)
    state: str = ""
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Substance":
        props = {name: Property.from_dict(name, p)
                 for name, p in payload.get("properties", {}).items()}
        if not props:
            raise ValueError(f"substance {payload.get('name')!r} has no properties")
        return cls(
            name=payload["name"],
            klass=payload["class"],
            properties=props,
            aliases=list(payload.get("aliases", [])),
            state=payload.get("state", ""),
            notes=payload.get("notes", ""),
        )

    @property
    def n_rows(self) -> int:
        return sum(p.n_rows for p in self.properties.values())

    def check(self) -> List[str]:
        problems: List[str] = []
        for prop in self.properties.values():
            problems.extend(prop.check())
        if not self.klass:
            problems.append(f"{self.name}: missing substance class")
        return problems


def validate_payload(payloads: List[Dict[str, Any]]) -> List[str]:
    """Validate a list of raw substance payloads; return all problems found."""
    problems: List[str] = []
    seen_classes = set()
    for raw in payloads:
        try:
            sub = Substance.from_dict(raw)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        seen_classes.add(sub.klass)
        problems.extend(sub.check())
        for alias in sub.aliases:
            if not alias.strip():
                problems.append(f"{sub.name}: blank alias")
    if not problems and not seen_classes:
        problems.append("no substances in payload")
    return problems