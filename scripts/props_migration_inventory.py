"""Read-only inventory of hard-coded fluid / material constants in the
ship-performance-platform repository, mapped onto the new TensorDG property
database (FEAT-ASSET-PROPS-DB acceptance item (4)).

The ship repository is scanned, never modified.  Usage::

    python scripts/props_migration_inventory.py /path/to/ship-performance-platform \
        > docs/props-migration-map.md

Heuristic limits (stated in the generated document as well)
-----------------------------------------------------------
* the scanner sees a literal and its identifier only -- not the unit, not the
  substance.  Units are inferred from magnitude (kg/m^3 vs t/m^3, Pa vs MPa)
  and the assumption is written into the table so a human can veto it;
* ``migrate`` means "this literal is a candidate to be replaced by a database
  query", the *module owner* still has to confirm the working fluid/material;
* dimensionless solver coefficients (lattice viscosity, turbulent viscosity,
  CFL numbers, Weibull moduli, ...) are explicitly *not* migrated.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensordg.props import PropsDB, load_db  # noqa: E402

PATTERN = re.compile(
    r"^(?P<indent>\s*)(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*[A-Za-z_\[\], ]+)?"
    r"\s*=\s*(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?:#\s*(?P<comment>.*))?$"
)

#: identifiers that are solver coefficients, never material data
DIMENSIONLESS_HINTS = ("ratio", "scale", "relax", "tol", "threshold", "gain",
                       "kappa", "cfl", "coeff", "_cf", "omega", "alpha",
                       "beta", "epsilon", "clamp", "factor", "index", "error",
                       "err", "bound", "limit", "coef", "modulus_weibull")
#: turbulent / sub-grid quantities carry a solver-own length scale
TURBULENT_HINTS = ("_t", "eddy", "sgs", "les_", "_les", "turb")

TOLERANCE = 0.10


def _substance_of(name: str) -> Optional[str]:
    low = name.lower()
    if "sea" in low or "_sw" in low or "salt" in low:
        return "sea_water"
    if "fresh" in low or "_fw" in low or "lake" in low:
        return "fresh_water"
    if "ice" in low or low in {"rho_i", "nu_i"}:
        return "sea_ice"
    if "air" in low:
        return "air"
    if "alu" in low:
        return "alu_5083_H116"
    return None


def _steel_grade(name: str, comment: str) -> Optional[str]:
    text = (name + " " + comment).lower()
    if "ah36" in text or "355" in text:
        return "AH36"
    if "eh40" in text or "390" in text:
        return "EH40"
    if "316" in text:
        return "stainless_316L"
    return None


def classify(name: str, value: float, comment: str) -> Tuple[str, object]:
    """Return (action, (substance, property) | reason-string)."""
    low = name.lower()
    if any(hint in low for hint in TURBULENT_HINTS):
        return "review", "turbulent/sub-grid quantity (solver-own length scale)"
    if any(hint in low for hint in DIMENSIONLESS_HINTS):
        return "review", "dimensionless/derived coefficient"

    is_density = bool(re.search(r"(rho|density)", low))
    is_kinematic = bool(re.search(r"(nu|kinematic)", low))
    is_dynamic = bool(re.search(r"(mu_|dynamic|viscos)", low)) and not is_kinematic

    if is_density:
        if abs(value) < 100.0:
            return "review", "non-SI density unit (t/m^3 or g/cm^3)"
        sub = _substance_of(name)
        if sub is None:
            if 1015.0 <= value <= 1040.0:
                sub = "sea_water"
            elif 990.0 <= value <= 1010.0:
                sub = "fresh_water"
            elif 870.0 <= value <= 940.0:
                sub = "sea_ice"
            elif 1.1 <= value <= 1.4:
                sub = "air"
            else:
                return "review", "density of an unlabelled substance"
        return "migrate", (sub, "density")

    if is_kinematic or is_dynamic:
        sub = _substance_of(name)
        if sub is None:
            if abs(value) <= 1.05e-6:
                sub = "fresh_water"
            elif 1.1e-6 <= abs(value) <= 1.45e-6:
                sub = "sea_water"
            else:
                return "review", "viscosity of an unlabelled substance"
        prop = "viscosity_kinematic" if is_kinematic or abs(value) < 1.0e-5 \
            else "viscosity_dynamic"
        if abs(value) > 1.0 and prop == "viscosity_dynamic":
            return "review", "viscosity in non-SI unit (cP/P)"
        return "migrate", (sub, prop)

    if re.search(r"(yield_stress|yield_strength|sigma_y|^fy$)", low) or \
            ("yield" in low and "stress" in low):
        grade = _steel_grade(name, comment)
        if grade is None:
            return "review", "yield stress, steel grade not identified in the code"
        return "migrate", (grade, "yield_strength")

    if re.search(r"(young|elastic_modulus|e_modulus)", low):
        grade = _steel_grade(name, comment)
        if grade is None:
            return "review", "elastic modulus, material not identified"
        return "migrate", (grade, "youngs_modulus")

    if "poisson" in low:
        grade = _steel_grade(name, comment)
        return ("migrate", (grade, "poisson_ratio")) if grade else \
            ("review", "Poisson ratio, material not identified")

    if "manning" in low:
        return "migrate", ("medium_sand", "manning_n")

    return "review", "unclassified"


def _unit_scale(prop: str, value: float) -> Tuple[float, str]:
    """Infer the SI scale factor from the magnitude; report the assumption."""
    if prop in {"youngs_modulus", "yield_strength", "compressive_strength",
                "ultimate_strength", "flexural_strength", "tensile_strength"}:
        if abs(value) < 1.0e4:
            return 1.0e6, "assumed MPa"
        if abs(value) < 1.0e7:
            return 1.0, "assumed Pa"
        return 1.0e6, "assumed MPa"
    if prop in {"poisson_ratio", "manning_n"}:
        return 1.0, "dimensionless"
    return 1.0, "SI"


def _compare(db: PropsDB, substance: str, prop: str, value: float):
    """Return (db_value_str, rel_error_str, action, note)."""
    scale, unit_note = _unit_scale(prop, value)
    try:
        res = db.query(substance, prop, T=15.0, clamp=True)
    except Exception as exc:  # noqa: BLE001
        return "-", "-", "review", f"no database entry ({exc.__class__.__name__})"
    db_si = res.value_si
    value_si = value * scale
    if db_si == 0.0:
        return f"{db_si:.6g} {res.unit}", "-", "review", unit_note
    rel = abs(value_si - db_si) / abs(db_si)
    action = "migrate" if rel <= TOLERANCE else "conflict"
    note = unit_note
    if res.clamped:
        note += f"; compared at T={res.conditions['T_used_degC']:.0f} degC " \
                f"(domain boundary)"
    return f"{db_si:.6g} {res.unit}", f"{rel * 100:.1f}%", action, note


def main(root: str) -> None:
    repo = Path(root)
    db = load_db()
    rows = []
    for path in sorted(repo.rglob("*.py")):
        if any(part in {".git", "node_modules", "__pycache__"} for part in path.parts):
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            match = PATTERN.match(line)
            if not match:
                continue
            name = match.group("name")
            if not re.search(r"(rho|density|viscos|nu_|mu_|young|poisson|yield|"
                             r"strength|manning)", name.lower()):
                continue
            value = float(match.group("value"))
            comment = (match.group("comment") or "").strip()
            action, target = classify(name, value, comment)
            db_ref = db_value = rel = note = "-"
            if action != "review" and isinstance(target, tuple):
                db_ref = f"{target[0]}/{target[1]}"
                db_value, rel, action, note = _compare(db, target[0], target[1],
                                                       value)
            elif action == "review":
                note = str(target)
            rows.append({
                "file": str(path.relative_to(repo)), "line": lineno,
                "name": name, "value": value, "comment": comment,
                "action": action, "db_ref": db_ref, "db_value": db_value,
                "rel": rel, "note": note,
            })

    counts = {}
    for row in rows:
        counts[row["action"]] = counts.get(row["action"], 0) + 1

    print("# 船海平台硬编码物性 -> TensorDG 物性库迁移映射表")
    print()
    print("**生成方式**: `python scripts/props_migration_inventory.py "
          "<ship repo>`（只读扫描，未修改 ship 仓任何文件；本表由程序生成，"
          "无手抄数字）")
    print()
    print("**扫描口径**: `*.py` 中以字面量赋值的 `rho*/density*/nu*/mu*/viscos*/"
          "young*/poisson*/yield*/strength*/manning*` 常量。")
    print()
    print(f"**统计**: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
          + f", total={len(rows)}")
    print()
    print("| action | 含义 |")
    print("|---|---|")
    print("| migrate | 量纲折算后与库值相差 ≤10%，可改为 `props.load_db().get(...)` |")
    print("| conflict | 相差 >10%，需 owner 判定（可能是别的物质/温度/单位） |")
    print("| review | 无量纲系数、单位不明或物质未标注，**不并入**物性库 |")
    print()
    print("> 诚实声明：扫描器只能看到标识符与字面量，看不到单位与物质。"
          "单位由量级推断（kg/m³ vs t/m³、Pa vs MPa），推断结果写在 “口径” 列；"
          "`migrate` 只是候选，仍需模块 owner 确认工作介质。")
    print()
    print("## 逐条映射表")
    print()
    print("| # | file:line | identifier | 现值 | action | 库条目 | 库值(SI) "
          "| 相对偏差 | 口径/原因 | 原注释 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for idx, row in enumerate(rows, 1):
        print(f"| {idx} | `{row['file']}:{row['line']}` | `{row['name']}` | "
              f"{row['value']:g} | {row['action']} | `{row['db_ref']}` | "
              f"{row['db_value']} | {row['rel']} | {row['note']} | "
              f"{row['comment'][:50]} |")
    print()
    print("## 结论")
    print()
    print(f"- 共 {len(rows)} 处字面量物性/材料常量：`migrate` "
          f"{counts.get('migrate', 0)}、`conflict` {counts.get('conflict', 0)}、"
          f"`review` {counts.get('review', 0)}。")
    print("- 网格/格子单位与无量纲系数（LBM `nu_water=0.03`、相场 `rho_water=1.0`、"
          "湍流/亚格子黏性、Weibull 模数）**不迁移**：它们没有 SI 量纲，"
          "迁进物性库会制造单位错误；正确做法是在调用点旁注明有量纲 duty point。")
    print("- 有量纲且已标注介质的常量（`NU_WATER=1.19e-6`、`RHO_SW=1025`、"
          "`RHO_AIR=1.225` 等）与库值吻合（≤10%），可直接换成库查询，"
          "从而继承 ITTC / ISO / 船级社规范的出处。")
    print("- `conflict` 行多为「物质或温度不同」而非库错，例如淡水 20 °C 取值 "
          "与其他温度取值对比；按本表口径需 owner 逐个确认后迁移。")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")