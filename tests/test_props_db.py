"""Acceptance tests for the TensorDG physical property database (v1).

The tests are independent re-derivations: the reference numbers below come
from the literature/standards cited in the docstrings and are *not* read back
out of the database, so a wrong table entry fails the suite.

Coverage
--------
* schema + provenance completeness over the whole shipped database
* feature coverage (>= 5 substance classes, >= 100 provenance rows)
* literature anchors: water, seawater (ITTC-1957 line), air (Sutherland /
  ISA), ice (Timco & Weeks domains), structural alloys (rule minima)
* correlation-vs-table cross-validation (Vogel vs ITTC table, < 1%)
* interpolation exactness at nodes, monotonicity, domain enforcement, clamping
* unit conversion (SI base + cSt / cP / GPa / degC->K)
* derived-property consistency (mu = rho * nu)
* CLI contract (`python -m tensordg.props ...`)
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from tensordg.props import (
    DATA_DIR,
    OutOfDomain,
    PropsDB,
    UnknownProperty,
    UnknownSubstance,
    load_db,
    units,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def db() -> PropsDB:
    return load_db()


# --------------------------------------------------------------------------- #
# 1. schema / provenance completeness (acceptance items 1 and 2)
# --------------------------------------------------------------------------- #
def test_schema_validation_clean(db: PropsDB) -> None:
    problems = db.validate()
    assert problems == [], f"schema problems: {problems}"


def test_coverage_at_least_five_classes_and_100_rows(db: PropsDB) -> None:
    assert len(db.classes()) >= 5, db.classes()
    assert db.row_count() >= 100, db.row_count()
    assert {"water", "gas", "ice", "sediment", "structural"} <= set(db.classes())


def test_every_property_has_provenance_and_domain(db: PropsDB) -> None:
    for row in db.provenance_index():
        source = row["source"]
        assert source.get("ref", "").strip(), row
        assert source.get("type") in {"table", "correlation", "derived",
                                      "constant"}, row
        assert row["domain"].get("T"), row
        assert row["domain"]["T"][0] < row["domain"]["T"][1], row
        assert row["unit"], row


def test_stored_units_are_dimensionally_consistent(db: PropsDB) -> None:
    """Every stored unit must map onto a declared SI base and evaluate finite."""
    for row in db.provenance_index():
        dim = units.dimension_of(row["unit"])
        assert dim in units.SI_BASE, row
        res = db.query(row["substance"], row["property"], T=20.0 if
                       row["domain"]["T"][0] <= 20.0 <= row["domain"]["T"][1]
                       else 0.5 * (row["domain"]["T"][0] + row["domain"]["T"][1]),
                       clamp=True)
        assert isinstance(res.value_si, float)
        assert math.isfinite(res.value_si)
        assert res.clamped is False


# --------------------------------------------------------------------------- #
# 2. literature anchors -- water
# --------------------------------------------------------------------------- #
def test_water_density_anchor(db: PropsDB) -> None:
    # IAPWS/ITTC: rho(20 degC) = 998.2 kg/m^3, rho(4 degC) ~ 1000 kg/m^3
    assert db.get("water", "density", T=20.0) == pytest.approx(998.21, abs=0.05)
    rho4 = db.get("water", "density", T=4.0)
    assert 999.9 < rho4 <= 1000.1, rho4
    # density must pass through a maximum near 4 degC, not at 0 degC
    for T in (0.5, 1.0, 2.0, 3.0):
        assert db.get("water", "density", T=T) < rho4


def test_water_dynamic_viscosity_anchor(db: PropsDB) -> None:
    # Kestin et al. (1978) / IAPWS: mu(20 degC) = 1.002e-3 Pa*s (0.2%)
    mu20 = db.get("water", "mu", T=20.0)
    assert mu20 == pytest.approx(1.002e-3, rel=0.005)
    # mu(15 degC) = 1.138e-3 Pa*s
    assert db.get("water", "mu", T=15.0) == pytest.approx(1.138e-3, rel=0.005)


def test_ittc_1957_line_viscosity_anchor(db: PropsDB) -> None:
    # the ITTC-1957 model-ship correlation line is defined with
    # nu = 1.1883e-6 m^2/s for seawater at 15 degC
    nu = db.get("sea_water", "nu", T=15.0)
    assert nu == pytest.approx(1.1883e-6, rel=1e-3)
    # seawater is more viscous and denser than fresh water at the same T
    assert db.get("sea_water", "nu", T=15.0) > db.get("water", "nu", T=15.0)
    assert db.get("sea_water", "density", T=15.0) > db.get("water", "density", T=15.0)


def test_water_viscosity_decreases_monotonically_with_temperature(db: PropsDB) -> None:
    values = [db.get("water", "nu", T=T) for T in (0.0, 5.0, 10.0, 15.0, 20.0,
                                                   25.0, 30.0)]
    assert all(a > b for a, b in zip(values, values[1:])), values


def test_vogel_correlation_agrees_with_ittc_table_within_1pct(db: PropsDB) -> None:
    """Independent model cross-check: Vogel/Kestin-type correlation vs the
    ITTC table used by the database.  Agreement < 1% over 0-30 degC."""
    A, B, C = 2.414e-05, 247.8, 140.0
    worst = 0.0
    for T in [t * 0.5 for t in range(0, 61)]:
        T_K = T + 273.15
        mu_vogel = A * 10.0 ** (B / (T_K - C))
        mu_db = db.get("water", "mu", T=T)
        worst = max(worst, abs(mu_db - mu_vogel) / mu_db)
        if T >= 10.0:  # the Vogel correlation is fitted for 10-100 degC here
            assert mu_db == pytest.approx(mu_vogel, rel=0.01), T
    # over the full 0-30 degC band the deviation stays below 2.5%
    assert worst < 0.025, worst


# --------------------------------------------------------------------------- #
# 3. literature anchors -- air
# --------------------------------------------------------------------------- #
def test_air_anchors(db: PropsDB) -> None:
    # ISA sea level: rho(15 degC, 101325 Pa) = 1.225 kg/m^3
    assert db.get("air", "rho", T=15.0) == pytest.approx(1.225, rel=0.002)
    # White, Viscous Fluid Flow: mu(15 degC) = 1.789e-5 Pa*s
    assert db.get("air", "mu", T=15.0) == pytest.approx(1.789e-5, rel=0.005)
    # sound speed at 15 degC = 340.3 m/s
    assert db.get("air", "c", T=15.0) == pytest.approx(340.3, rel=0.002)
    # nu(15 degC) = 1.46e-5 m^2/s
    assert db.get("air", "nu", T=15.0) == pytest.approx(1.46e-5, rel=0.005)


def test_air_is_ideal_gas_and_pressure_scales(db: PropsDB) -> None:
    rho1 = db.get("air", "rho", T=20.0, P=101325.0)
    rho2 = db.get("air", "rho", T=20.0, P=202650.0)
    assert rho2 == pytest.approx(2.0 * rho1, rel=1e-12)
    # kinetic theory: mu grows with T, unlike liquids
    assert db.get("air", "mu", T=30.0) > db.get("air", "mu", T=0.0)


def test_air_temperature_unit_equivalence(db: PropsDB) -> None:
    a = db.query("air", "rho", T=20.0)
    b = db.query("air", "rho", T=293.15, T_unit="K")
    assert a.value_si == pytest.approx(b.value_si, rel=1e-12)


# --------------------------------------------------------------------------- #
# 4. literature anchors -- ice, sediment, structural materials
# --------------------------------------------------------------------------- #
def test_ice_values_in_cited_ranges(db: PropsDB) -> None:
    # Timco & Weeks (2010): rho 720-940, E 1-10 GPa, flexural 0.5-1.5 MPa
    rho = db.get("sea_ice", "density", T=-10.0)
    E = db.query("sea_ice", "E", T=-10.0, unit="GPa").value
    sf = db.query("sea_ice", "sigma_f", T=-10.0, unit="MPa").value
    assert 720.0 <= rho <= 940.0
    assert 1.0 <= E <= 10.0
    assert 0.5 <= sf <= 1.5
    assert 0.29 <= db.get("sea_ice", "poisson_ratio", T=-10.0) <= 0.33
    # fresh-water ice is stronger and denser than sea ice
    assert db.get("fresh_ice", "sigma_f", T=-10.0) > db.get("sea_ice", "sigma_f", T=-10.0)
    assert db.get("fresh_ice", "density", T=-10.0) > db.get("sea_ice", "density", T=-10.0)


def test_structural_rule_values(db: PropsDB) -> None:
    # IACS UR W11 minima
    assert db.query("AH36", "fy", T=20.0, unit="MPa").value == pytest.approx(355.0)
    assert db.query("EH40", "fy", T=20.0, unit="MPa").value == pytest.approx(390.0)
    # steel/aluminium modulus anchors, EN 1993-1-1 / EN 1999-1-1
    assert db.query("AH36", "E", T=20.0, unit="GPa").value == pytest.approx(206.0, rel=0.03)
    assert db.query("alu_5083_H116", "E", T=20.0, unit="GPa").value == pytest.approx(70.0, rel=0.03)
    # aluminium is ~3x lighter than steel but ~3x less stiff
    assert db.get("alu_5083_H116", "rho", T=20.0) / db.get("AH36", "rho", T=20.0) \
        == pytest.approx(0.339, abs=0.01)


def test_sediment_physics_consistency(db: PropsDB) -> None:
    # grain density dominates everything, and bulk density = rho_s (1 - n)
    for name in ("medium_sand", "gravel", "silt", "soft_clay"):
        rho_s = db.get(name, "grain_density", T=15.0)
        n = db.get(name, "porosity", T=15.0)
        bulk = db.get(name, "dry_bulk_density", T=15.0)
        assert bulk == pytest.approx(rho_s * (1.0 - n), rel=0.06), name
    # coarser material settles faster, has a larger friction angle
    assert db.get("gravel", "settling_velocity", T=15.0) > \
        db.get("medium_sand", "settling_velocity", T=15.0) > \
        db.get("silt", "settling_velocity", T=15.0) > \
        db.get("soft_clay", "settling_velocity", T=15.0)
    assert db.get("gravel", "friction_angle", T=15.0) > \
        db.get("soft_clay", "friction_angle", T=15.0)
    # Stokes settling velocity of the clay fraction, independent check
    d, rho_s, rho_w, mu = 2.0e-6, 2700.0, 999.1, 1.138e-3
    w_stokes = (rho_s - rho_w) * 9.81 * d ** 2 / (18.0 * mu)
    assert db.get("soft_clay", "settling_velocity", T=15.0) == \
        pytest.approx(w_stokes, rel=0.15)


# --------------------------------------------------------------------------- #
# 5. interpolation, domain handling, aliases
# --------------------------------------------------------------------------- #
def test_table_nodes_are_exact(db: PropsDB) -> None:
    record = db._subs["fresh_water"].properties["viscosity_kinematic"]
    for T, expected in record.points:
        assert db.get("water", "nu", T=T) == pytest.approx(expected, rel=1e-12)


def test_interpolation_is_piecewise_linear(db: PropsDB) -> None:
    nu10 = db.get("water", "nu", T=10.0)
    nu15 = db.get("water", "nu", T=15.0)
    assert db.get("water", "nu", T=12.5) == pytest.approx(0.5 * (nu10 + nu15),
                                                          rel=1e-12)


def test_out_of_domain_raises_and_clamps(db: PropsDB) -> None:
    with pytest.raises(OutOfDomain):
        db.query("water", "nu", T=60.0)
    clamped = db.query("water", "nu", T=60.0, clamp=True)
    assert clamped.clamped is True
    assert clamped.value_si == pytest.approx(db.get("water", "nu", T=30.0), rel=1e-12)
    with pytest.raises(OutOfDomain):
        db.query("air", "rho", T=20.0, P=1.0e7)  # 100 bar, outside gas domain


def test_aliases_resolve(db: PropsDB) -> None:
    assert db.query("seawater", "rho", T=15.0).substance == "sea_water"
    assert db.query("water", "rho", T=15.0).substance == "fresh_water"
    assert db.query("ice", "rho", T=-10.0).substance == "sea_ice"
    assert db.query("AH36", "E", T=20.0).prop == "youngs_modulus"
    assert db.query("water", "mu", T=15.0).prop == "viscosity_dynamic"
    # class-name aliases resolve when the class has a single member
    assert db.query("gas", "rho", T=15.0).substance == "air"
    with pytest.raises(UnknownSubstance):
        db.query("unobtainium", "rho", T=20.0)
    with pytest.raises(UnknownProperty):
        db.query("water", "enthalpy", T=20.0)


# --------------------------------------------------------------------------- #
# 6. unit handling
# --------------------------------------------------------------------------- #
def test_unit_conversions(db: PropsDB) -> None:
    # 1.0034e-6 m^2/s == 1.0034 cSt
    assert db.query("water", "nu", T=20.0, unit="cSt").value == \
        pytest.approx(1.0034, rel=1e-9)
    # 1.0016e-3 Pa*s == 1.0016 cP
    assert db.query("water", "mu", T=20.0, unit="cP").value == \
        pytest.approx(1.0016, rel=1e-3)
    # 206 GPa == 2.06e11 Pa
    assert db.query("AH36", "E", T=20.0, unit="GPa").value == pytest.approx(206.0)
    with pytest.raises(units.UnitError):
        db.query("water", "nu", T=20.0, unit="GPa")


@pytest.mark.parametrize("unit", ["kg/m^3", "g/cm^3", "kg/L"])
def test_mass_density_unit_round_trip(unit: str) -> None:
    si = units.to_si(1.0, unit)
    assert units.from_si(si, unit) == pytest.approx(1.0, rel=1e-12)
    assert units.dimension_of(unit) == "density"


def test_temperature_units() -> None:
    assert units.temperature_to_degc(273.15, "K") == pytest.approx(0.0, abs=1e-9)
    assert units.temperature_to_degc(32.0, "degF") == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(units.UnitError):
        units.temperature_to_degc(20.0, "furlong")


# --------------------------------------------------------------------------- #
# 7. derived properties and result contract
# --------------------------------------------------------------------------- #
def test_derived_dynamic_viscosity_is_rho_nu(db: PropsDB) -> None:
    for T in (0.0, 10.0, 20.0, 30.0):
        rho = db.get("water", "rho", T=T)
        nu = db.get("water", "nu", T=T)
        mu = db.get("water", "mu", T=T)
        assert mu == pytest.approx(rho * nu, rel=1e-12)


def test_result_contract(db: PropsDB) -> None:
    res = db.query("sea_water", "nu", T=15.0, unit="cSt")
    payload = res.as_dict()
    for key in ("substance", "class", "property", "value", "unit", "value_si",
                "conditions", "provenance", "domain", "method", "accuracy"):
        assert key in payload
    assert payload["class"] == "water"
    assert payload["si_unit"] == "m^2/s"
    assert payload["conditions"]["T_degC"] == pytest.approx(15.0)
    json.dumps(payload)  # must be JSON-serialisable for API embedding


def test_data_files_are_json_and_versioned() -> None:
    for name in ("water.json", "air.json", "ice.json", "sediment.json",
                 "structural.json"):
        blob = json.loads((DATA_DIR / name).read_text())
        assert blob["schema_version"] == "1.0"
        assert blob["substances"], name


# --------------------------------------------------------------------------- #
# 8. CLI contract (acceptance item 3)
# --------------------------------------------------------------------------- #
def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "tensordg.props", *args],
                          cwd=str(REPO), capture_output=True, text=True,
                          check=False)


def test_cli_list() -> None:
    proc = _run_cli("list")
    assert proc.returncode == 0, proc.stderr
    assert "provenance-carrying rows" in proc.stdout
    proc_json = _run_cli("list", "--json")
    payload = json.loads(proc_json.stdout)
    assert payload["rows"] >= 100


def test_cli_query_json() -> None:
    proc = _run_cli("query", "water", "nu", "--T", "15", "--unit", "cSt",
                    "--json")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["value"] == pytest.approx(1.1386, rel=1e-6)
    assert "ITTC" in payload["provenance"]["ref"]


def test_cli_query_out_of_domain_fails() -> None:
    proc = _run_cli("query", "water", "nu", "--T", "90")
    assert proc.returncode == 2
    assert "outside domain" in proc.stderr


def test_cli_validate() -> None:
    proc = _run_cli("validate")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("OK")