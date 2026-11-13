"""Bind ``docs/CASE-ACOUSTIC-PULSE.md`` to its machine record *and* to the solver.

Why this test exists
--------------------
The acoustic-pulse case is only worth quoting if three things agree:
the document, the artifact it calls its machine record
(``validation/acoustic_pulse/convergence.json``), and the code that regenerates
that record.  Nothing in the repository checked that triangle, so the failure
mode is the one that already happened once for the Ritter dry-bed number: a
documented figure silently drifts away from its own declared source of truth and
no test fails (see ``tests/test_evidence_integrity.py``).

This module closes the triangle at the printed precision of the document:

1. **Document -> record.**  Every cell of the design-order table, the full
   convergence record table, the "error judgement" table (§4) and the
   field-comparison claims (§3.2) must equal the record value rounded to the
   digits the document prints -- no re-typing, and a refreshed record that the
   document does not follow fails here.
2. **Record -> solver.**  The two cheapest rungs of the ladder are re-run *from
   the solver* and must reproduce the committed record to ``1e-9`` relative.
   A change in ``tensordg.acoustics`` or in the benchmark driver that moves a
   number invalidates the record instead of shipping a stale claim.
3. **Record -> figures.**  The figures the record declares exist, are real PNGs
   and are the two the document embeds.

Run: ``python -m pytest tests/test_acoustic_pulse_doc_binding.py -q``
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "docs" / "CASE-ACOUSTIC-PULSE.md"
RECORD_PATH = ROOT / "validation" / "acoustic_pulse" / "convergence.json"

# Gate quoted in §4 of the document: the domain's unacceptable error bar.
DOCUMENTED_ERROR_BAR = 0.15
ABSOLUTE_CLAIM_BAR = 0.03          # the "< 3%?" column
SOLVER_REPRO_RTOL = 1e-9


def _ulp(text: str) -> float:
    """Unit in the last printed digit of a decimal/scientific literal."""
    text = text.strip()
    mantissa, _, exponent = text.lower().partition("e")
    exponent = int(exponent) if exponent else 0
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    return 10.0 ** (exponent - decimals)


def assert_printed(actual: float, text: str, label: str) -> None:
    """``actual`` must equal ``text`` to the precision ``text`` is printed at."""
    tol = _ulp(text)
    claimed = float(text)
    message = f"{label}: document says {text}, record says {actual!r} (tol {tol:g})"
    assert abs(actual - claimed) <= tol, message


def _doc_text() -> str:
    return DOC_PATH.read_text(encoding="utf-8")


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


@pytest.fixture(scope="module")
def record() -> dict:
    assert RECORD_PATH.exists(), f"missing machine record {RECORD_PATH}"
    return json.loads(RECORD_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def doc() -> str:
    assert DOC_PATH.exists(), f"missing case document {DOC_PATH}"
    return _doc_text()


def _design_order_rows(doc: str) -> dict[tuple[int, int, int], dict]:
    rows = {}
    for line in doc.splitlines():
        if not line.startswith("|"):
            continue
        cells = _table_cells(line)
        if len(cells) != 4 or not re.fullmatch(r"\d+\s*\u2192\s*\d+\^2", cells[1]):
            continue
        coarse, fine = (int(part) for part in
                        re.findall(r"(\d+)\s*\u2192\s*(\d+)\^2", cells[1])[0])
        key = (int(cells[0]), coarse, fine)
        assert key not in rows, f"duplicate design-order row {key}"
        rows[key] = {"observed_order": cells[2], "target_order": cells[3]}
    return rows


def _full_record_rows(doc: str) -> dict[tuple[int, int], dict]:
    rows = {}
    for line in doc.splitlines():
        if not line.startswith("|"):
            continue
        cells = _table_cells(line)
        # The header of this table contains a stray pipe inside "|E-E0|/E0",
        # so data rows are identified structurally: 8 cells, degree + cells.
        if len(cells) != 8 or not cells[0].isdigit() or not cells[1].isdigit():
            continue
        key = (int(cells[0]), int(cells[1]))
        assert key not in rows, f"duplicate record-table row {key}"
        rows[key] = {
            "h": cells[2], "steps": cells[3],
            "relative_l2_state": cells[4], "relative_l2_pressure": cells[5],
            "energy": cells[6], "energy_drift": cells[7],
        }
    return rows


def _judgement_rows(doc: str) -> list[list[str]]:
    rows = []
    for line in doc.splitlines():
        if not line.startswith("|"):
            continue
        cells = _table_cells(line)
        if len(cells) != 5 or cells[0] == "quantity":
            continue
        if set(cells[0]) <= set("-: "):     # the |---|---| separator
            continue
        rows.append(cells)
    return rows


# --------------------------------------------------------------------------
# 1. the record itself is a passing ladder
# --------------------------------------------------------------------------
def test_record_is_a_complete_passing_ladder(record):
    assert record["benchmark"] == "acoustic_pulse_2d"
    assert record["case"] == "axisymmetric_gaussian_pulse"
    assert record["passed"] is True
    for name, gate in record["gates"].items():
        assert gate["pass"] is True, f"record gate {name} did not pass"
    assert {row["degree"] for row in record["rows"]} == {1, 2, 3}
    assert {row["cells"] for row in record["rows"]} == {16, 32, 64}
    assert len(record["rows"]) == 9
    assert len(record["orders"]) == 6
    assert all(item["pass"] for item in record["orders"])
    # the documented 15 % bar must be honoured with a wide margin
    finest = [row for row in record["rows"] if row["cells"] == 64]
    for row in finest:
        assert row["relative_l2_state"] < ABSOLUTE_CLAIM_BAR < DOCUMENTED_ERROR_BAR


# --------------------------------------------------------------------------
# 2. parameter table -> record
# --------------------------------------------------------------------------
def test_parameter_table_matches_the_record(doc, record):
    flat = re.sub(r"\s+", " ", doc)
    params = {}
    for line in doc.splitlines():
        if line.startswith("|"):
            cells = _table_cells(line)
            if len(cells) == 2:
                params[cells[0]] = cells[1]

    def first_number(key: str) -> float:
        return float(re.search(r"-?\d+(?:\.\d+)?", params[key]).group(0))

    assert first_number("`sigma`") == pytest.approx(record["sigma"], abs=1e-15)
    assert first_number("final time") == pytest.approx(record["final_time"], abs=1e-15)
    assert first_number("CFL") == pytest.approx(record["cfl"], abs=1e-15)

    degrees = [int(d) for d in re.findall(r"\d+", params["polynomial degrees"])]
    assert degrees == sorted({row["degree"] for row in record["rows"]})
    meshes = [int(m) for m in re.findall(r"(\d+)\^2", params["meshes"])]
    assert meshes == sorted({row["cells"] for row in record["rows"]})

    rho, c = re.findall(r"=\s*(\d+(?:\.\d+)?)", params["medium"])[:2]
    assert float(rho) == record["medium"]["rho"]
    assert float(c) == record["medium"]["c"]
    assert record["medium"]["bulk_modulus"] == pytest.approx(
        record["medium"]["rho"] * record["medium"]["c"] ** 2, rel=1e-15)

    domain = re.search(r"\[0,(\d+(?:\.\d+)?)\] x \[0,(\d+(?:\.\d+)?)\]", params["domain"])
    assert float(domain.group(1)) == record["length"] == float(domain.group(2))
    assert "periodic" in flat  # the scope limit stated in §5 must stay stated


# --------------------------------------------------------------------------
# 3. design-order table -> record (bijection)
# --------------------------------------------------------------------------
def test_design_order_table_is_a_bijection_with_the_record(doc, record):
    rows = _design_order_rows(doc)
    by_key = {(item["degree"], item["from_cells"], item["to_cells"]): item
              for item in record["orders"]}
    assert set(rows) == set(by_key), (
        f"document rows {sorted(rows)} != record rows {sorted(by_key)}")
    for key, cells in rows.items():
        item = by_key[key]
        assert_printed(item["observed_order"], cells["observed_order"],
                       f"observed order {key}")
        assert int(cells["target_order"]) == item["target_order"], key
        # the documented number has to reproduce the recorded pass verdict
        assert item["pass"] is True


# --------------------------------------------------------------------------
# 4. full convergence table -> record (bijection)
# --------------------------------------------------------------------------
def test_full_record_table_is_a_bijection_with_the_record(doc, record):
    rows = _full_record_rows(doc)
    by_key = {(row["degree"], row["cells"]): row for row in record["rows"]}
    assert set(rows) == set(by_key), (
        f"document rows {sorted(rows)} != record rows {sorted(by_key)}")
    for key, cells in rows.items():
        row = by_key[key]
        assert_printed(row["h"], cells["h"], f"h {key}")
        assert int(cells["steps"]) == row["steps"], f"steps {key}"
        assert_printed(row["relative_l2_state"], cells["relative_l2_state"],
                       f"relative L2 state {key}")
        assert_printed(row["relative_l2_pressure"], cells["relative_l2_pressure"],
                       f"relative L2 pressure {key}")
        assert_printed(row["energy"], cells["energy"], f"energy {key}")
        assert_printed(row["energy_drift"], cells["energy_drift"],
                       f"energy drift {key}")


# --------------------------------------------------------------------------
# 5. §4 error judgement -> record, including the "< 3%?" verdicts
# --------------------------------------------------------------------------
def test_error_judgement_table_binds_to_the_record(doc, record):
    by_key = {(row["degree"], row["cells"]): row for row in record["rows"]}
    rows = _judgement_rows(doc)
    assert len(rows) == 6, f"expected 6 judgement rows, found {len(rows)}"

    for cells in rows:
        claim, measured = cells[0], cells[2]
        match = re.search(r"`?p\s*=\s*(\d),\s*(\d+)\^2", claim)
        if measured[0].isdigit() and match:
            degree, cells_n = int(match.group(1)), int(match.group(2))
            row = by_key[(degree, cells_n)]
            field = ("energy_drift" if claim.startswith("energy drift")
                     else "relative_l2_state")
            assert_printed(row[field], measured, f"judgement {claim}")
            assert (row[field] < ABSOLUTE_CLAIM_BAR) == ("yes" in cells[4]), (
                f"verdict in the '< 3%?' column disagrees with the record: "
                f"{claim} -> {row[field]:.3e}")
            if field == "relative_l2_state":
                percent = re.search(r"\(([\d.]+)%\)", cells[4])
                assert percent, f"missing percentage in {cells[4]}"
                assert_printed(row[field] * 100.0, percent.group(1),
                               f"percentage {claim}")

    orders_cell = next(cells for cells in rows
                       if cells[0].startswith("convergence order"))
    documented = [value.strip() for value in orders_cell[2].split(",")]
    finest_orders = {(item["degree"], item["from_cells"], item["to_cells"]): item
                     for item in record["orders"] if item["to_cells"] == 64}
    assert len(documented) == len(finest_orders) == 3
    # the §4 list quotes the finest-pair order of each degree, in degree order
    for key, text in zip(sorted(finest_orders), documented):
        assert_printed(finest_orders[key]["observed_order"], text,
                       f"finest observed order {key}")

    # the declared order deficit must stay consistent with the numbers
    deficit = re.search(r"\u2264\s*([\d.]+)", orders_cell[3])
    assert deficit, f"cannot parse the order-deficit claim from {orders_cell[3]}"
    worst = max(item["target_order"] - item["observed_order"]
                for item in record["orders"])
    assert worst <= float(deficit.group(1)) + 1e-12, (
        f"record violates the documented order deficit: {worst:.4f} > {deficit.group(1)}")


# --------------------------------------------------------------------------
# 6. §3.2 field comparison and the energy anchor
# --------------------------------------------------------------------------
def test_field_comparison_claims_bind_to_the_record(doc, record):
    flat = re.sub(r"\s+", " ", doc)
    metrics = record["figure_metrics"]
    assert metrics, "record has no figure_metrics: regenerating without figures rots §3.2"
    match = re.search(
        r"maximum pointwise pressure error is \*\*([\d.eE+-]+)\*\* against a field "
        r"scale of ([\d.]+) \(([\d.eE+-]+) %\)", flat)
    assert match, "cannot find the §3.2 field-comparison claim"
    assert_printed(metrics["field_max_abs_error"], match.group(1), "field max |p| error")
    assert_printed(metrics["field_scale"], match.group(2), "field scale")
    relative_percent = metrics["field_max_abs_error"] / metrics["field_scale"] * 100.0
    assert_printed(relative_percent, match.group(3), "field error percentage")

    finest = max(record["rows"], key=lambda row: (row["degree"], row["cells"]))
    # §3.2 attributes the field figure to the highest-degree finest run
    assert (finest["degree"], finest["cells"]) == (3, 64)


def test_energy_anchor_claim_binds_to_the_record(doc, record):
    flat = re.sub(r"\s+", " ", doc)
    match = re.search(r"E0 = pi sigma\^2/\(4K\) = ([\d.]+)", flat)
    assert match, "cannot find the analytic initial-energy claim"
    assert_printed(record["energy_exact_initial"], match.group(1), "E0")
    # the closed form quoted in the document is the one the record carries
    import math
    closed_form = math.pi * record["sigma"] ** 2 / (4.0 * record["medium"]["bulk_modulus"])
    assert record["energy_exact_initial"] == pytest.approx(closed_form, rel=1e-15)


# --------------------------------------------------------------------------
# 7. declared figures exist and are the ones the document embeds
# --------------------------------------------------------------------------
def test_declared_figures_exist_and_are_png(doc, record):
    figures = record["figures"]
    assert len(figures) == 2, f"record declares {figures}"
    for relative in figures:
        path = ROOT / relative
        assert path.exists(), f"declared figure missing: {relative}"
        payload = path.read_bytes()
        assert payload.startswith(b"\x89PNG\r\n\x1a\n"), f"not a PNG: {relative}"
        assert len(payload) > 20_000, f"suspiciously small figure: {relative}"
        assert path.name in doc, f"figure {relative} is not embedded in the document"


# --------------------------------------------------------------------------
# 8. record -> solver: the two cheapest rungs must reproduce
# --------------------------------------------------------------------------
@pytest.mark.parametrize("degree,cells", [(1, 16), (2, 16)])
def test_cheapest_rungs_reproduce_from_the_solver(record, degree, cells):
    from benchmarks.acoustic_pulse_2d import run_point
    from tensordg import AcousticMedium

    row = next(item for item in record["rows"]
               if item["degree"] == degree and item["cells"] == cells)
    medium = AcousticMedium(rho=record["medium"]["rho"], c=record["medium"]["c"])
    fresh = run_point(cells, degree, final_time=record["final_time"],
                      sigma=record["sigma"], length=record["length"],
                      cfl=record["cfl"], medium=medium)
    assert fresh["steps"] == row["steps"], (
        f"P{degree} {cells}^2 step count moved: {fresh['steps']} != {row['steps']}")
    for field in ("relative_l2_state", "relative_l2_pressure", "energy",
                  "energy_drift"):
        assert fresh[field] == pytest.approx(row[field], rel=SOLVER_REPRO_RTOL), (
            f"P{degree} {cells}^2 {field}: solver now gives {fresh[field]!r}, "
            f"record still claims {row[field]!r} -- refresh validation/acoustic_pulse")
    # the drift of the fresh run must still satisfy the recorded gate where the
    # record applies it (the coarse grids legitimately dissipate, see §3.1/§5)
    if cells == max(row["cells"] for row in record["rows"]):
        assert fresh["energy_drift"] <= record["gates"]["energy"]["tolerance"]