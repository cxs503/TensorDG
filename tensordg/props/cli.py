"""Command line interface for the TensorDG property database.

    python -m tensordg.props list
    python -m tensordg.props info sea_water
    python -m tensordg.props query water nu --T 15 --unit cSt
    python -m tensordg.props query air mu --T 20 --unit cP --json
    python -m tensordg.props provenance water density
    python -m tensordg.props validate
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from .db import PropsDB, PropsError, load_db


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def cmd_list(db: PropsDB, args: argparse.Namespace) -> int:
    rows = db.summary()
    if args.json:
        print(json.dumps({"meta": db.meta, "rows": db.row_count(),
                          "substances": rows}, indent=2))
        return 0
    print(f"{'substance':22s} {'class':12s} {'rows':>5s}  properties")
    for row in rows:
        print(f"{row['substance']:22s} {row['class']:12s} {row['rows']:5d}  "
              f"{', '.join(row['properties'])}")
    print(f"\n{len(rows)} substances, {len(db.classes())} classes, "
          f"{db.row_count()} provenance-carrying rows")
    return 0


def cmd_info(db: PropsDB, args: argparse.Namespace) -> int:
    props = db.provenance_index()
    selected = [p for p in props if p["substance"] == args.substance]
    if not selected:
        names = db.substances()
        print(f"unknown substance {args.substance!r}; known: {names}",
              file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(selected, indent=2))
        return 0
    for row in selected:
        dom = ", ".join(f"{k}: [{v[0]}, {v[1]}]" for k, v in row["domain"].items())
        print(f"{row['substance']}/{row['property']}  [{row['unit']}]  "
              f"rows={row['rows']}")
        print(f"    domain  : {dom}")
        print(f"    source  : {row['source'].get('ref')}")
        if row["source"].get("note"):
            print(f"    note    : {row['source']['note']}")
        if row["accuracy"]:
            print(f"    accuracy: {row['accuracy']}")
    return 0


def cmd_query(db: PropsDB, args: argparse.Namespace) -> int:
    try:
        res = db.query(args.substance, args.property, T=args.T,
                       P=args.P, T_unit=args.T_unit, unit=args.unit,
                       clamp=args.clamp)
    except PropsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(res.as_dict(), indent=2))
        return 0
    cond = ", ".join(f"{k}={_fmt(v)}" for k, v in res.conditions.items()
                     if k != "requested_T_unit")
    print(f"{res.substance} / {res.prop}")
    print(f"  conditions : {cond}")
    print(f"  value      : {_fmt(res.value)} {res.unit}")
    print(f"  SI         : {_fmt(res.value_si)} "
          f"{_si_unit(res.unit)}")
    print(f"  method     : {res.method}")
    print(f"  source     : {res.source.get('ref')}")
    print(f"  accuracy   : {res.accuracy}")
    if res.clamped:
        print("  warning    : condition clamped to the domain boundary")
    return 0


def cmd_provenance(db: PropsDB, args: argparse.Namespace) -> int:
    try:
        res = db.query(args.substance, args.property,
                       T=args.T if args.T is not None else 20.0,
                       T_unit=args.T_unit, clamp=True)
    except PropsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    payload: Dict[str, Any] = {
        "substance": res.substance,
        "property": res.prop,
        "unit": res.unit,
        "domain": res.domain,
        "source": res.source,
        "accuracy": res.accuracy,
        "applicability": res.applicability,
    }
    print(json.dumps(payload, indent=2))
    return 0


def cmd_validate(db: PropsDB, args: argparse.Namespace) -> int:
    problems: List[str] = db.validate()
    if problems:
        print(f"{len(problems)} schema problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"OK: {len(db.substances())} substances, {db.row_count()} rows, "
          f"all with provenance and a declared domain")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tensordg.props",
        description="TensorDG physical property database (SI base, "
                    "provenance-carrying)")
    parser.add_argument("--data-dir", default=None,
                        help="directory with the JSON property files")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list substances and property count")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_info = sub.add_parser("info", help="show all properties of a substance")
    p_info.add_argument("substance")
    p_info.add_argument("--json", action="store_true")
    p_info.set_defaults(func=cmd_info)

    p_query = sub.add_parser("query", help="evaluate one property")
    p_query.add_argument("substance")
    p_query.add_argument("property")
    p_query.add_argument("--T", type=float, default=None,
                         help="temperature on the property axis (default degC)")
    p_query.add_argument("--T-unit", dest="T_unit", default="degC")
    p_query.add_argument("--P", type=float, default=101325.0,
                         help="absolute pressure in Pa (gas models)")
    p_query.add_argument("--unit", default=None,
                         help="output unit, e.g. cSt / cP / GPa / MPa")
    p_query.add_argument("--clamp", action="store_true",
                         help="clamp to the domain boundary instead of raising")
    p_query.add_argument("--json", action="store_true")
    p_query.set_defaults(func=cmd_query)

    p_prov = sub.add_parser("provenance", help="print the citation of a property")
    p_prov.add_argument("substance")
    p_prov.add_argument("property")
    p_prov.add_argument("--T", type=float, default=None)
    p_prov.add_argument("--T-unit", dest="T_unit", default="degC")
    p_prov.set_defaults(func=cmd_provenance)

    p_val = sub.add_parser("validate", help="run schema + provenance checks")
    p_val.set_defaults(func=cmd_validate)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    db = load_db(args.data_dir)
    return int(args.func(db, args))


def _si_unit(unit: str) -> str:
    from .units import SI_BASE, dimension_of

    return SI_BASE.get(dimension_of(unit), "?")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())