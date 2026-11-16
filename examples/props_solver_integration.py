"""Solver integration samples for the TensorDG property database (v1).

Four end-to-end consumers, each one showing that the database replaces
hard-coded fluid/material constants inside a solver set-up path:

A. Resistance / model-scale hydrodynamics
   seawater viscosity -> Reynolds number -> ITTC-1957 friction coefficient,
   including the model/full-scale scale effect.
B. Compressible DG solver (``tensordg.euler_dg_2d``)
   air density / sound speed -> conservative initial state for a Mach 0.5
   free stream, and the CFL time step that follows from it.
C. Shallow-water solver (``tensordg.swe_1d`` / ``swe_2d``)
   sediment Manning coefficient -> bed-friction velocity used by the source
   term.
D. Structural / ice-class screening
   steel modulus + Poisson ratio -> DNV-RP-C201 plate buckling stress, and
   first-year ice flexural strength -> ISO 19906 flexural failure load.

Run:  python examples/props_solver_integration.py
"""

from __future__ import annotations

import math

from tensordg.props import load_db


# --------------------------------------------------------------------------- #
# A. resistance: Re and ITTC-1957 friction line
# --------------------------------------------------------------------------- #
def ittc57_friction(Re: float) -> float:
    """ITTC-1957 model-ship correlation line C_F = 0.075/(log10 Re - 2)^2."""
    return 0.075 / (math.log10(Re) - 2.0) ** 2


def resistance_sample(db, L_pp: float = 230.0, U: float = 12.0,
                      T: float = 15.0) -> dict:
    nu = db.get("sea_water", "nu", T=T)          # ITTC 7.5-02-01-03 table
    rho = db.get("sea_water", "density", T=T)
    Re = U * L_pp / nu
    Cf = ittc57_friction(Re)
    # model at 1:40 scale, same temperature (ITTC standard condition)
    L_m = L_pp / 40.0
    U_m = U / math.sqrt(40.0)                    # Froude similarity
    Re_m = U_m * L_m / nu
    return {
        "T_degC": T, "nu": nu, "rho": rho,
        "Re_ship": Re, "Cf_ship": Cf,
        "Re_model": Re_m, "Cf_model": ittc57_friction(Re_m),
        "Cf_ratio": ittc57_friction(Re_m) / Cf,
        "nu_provenance": db.query("sea_water", "nu", T=T).source["ref"],
    }


# --------------------------------------------------------------------------- #
# B. compressible DG initial state from air properties
# --------------------------------------------------------------------------- #
def compressible_state(db, mach: float = 0.5, p0: float = 101325.0,
                       T: float = 20.0) -> dict:
    rho = db.get("air", "rho", T=T, P=p0)
    mu = db.get("air", "mu", T=T)
    a = db.get("air", "c", T=T)
    u = mach * a
    gamma = 1.4
    E = p0 / (gamma - 1.0) + 0.5 * rho * u * u
    # explicit-DG CFL step for a P1 scheme on a 0.1 m mesh
    h, p_order = 0.1, 1
    dt = 0.1 * h / ((2 * p_order + 1) * (abs(u) + a))
    return {"rho": rho, "u": u, "p": p0, "E": E, "mu": mu, "a": a,
            "Re_per_m": rho * u / mu, "dt_cfl": dt}


# --------------------------------------------------------------------------- #
# C. shallow-water bed friction from the sediment Manning coefficient
# --------------------------------------------------------------------------- #
def manning_bed_friction(db, depth: float = 2.0, slope: float = 1e-3,
                         sediment: str = "medium_sand", T: float = 15.0) -> dict:
    n = db.get(sediment, "manning_n", T=T)
    # Manning: U = (1/n) R^(2/3) S^(1/2) with R ~= depth for a wide channel
    U = (1.0 / n) * depth ** (2.0 / 3.0) * math.sqrt(slope)
    # equivalent quadratic drag coefficient for an SWE source term
    Cd = 9.81 * n * n / depth ** (1.0 / 3.0)
    return {"sediment": sediment, "manning_n": n, "U": U, "Cd": Cd,
            "tau_b_Pa": db.get("fresh_water", "density", T=T) * Cd * U * U}


# --------------------------------------------------------------------------- #
# D. structural / ice screening
# --------------------------------------------------------------------------- #
def plate_buckling(db, material: str = "AH36", t: float = 0.010,
                   b: float = 0.800) -> dict:
    """Uniaxial plate buckling stress, DNV-RP-C201 / EN 1993-1-5 form."""
    E = db.get(material, "E", T=20.0)
    nu = db.get(material, "poisson_ratio", T=20.0)
    sigma_cr = math.pi ** 2 * E / (12.0 * (1.0 - nu ** 2)) * (t / b) ** 2
    # composites have no yield point -- fall back to the compressive strength
    try:
        strength = db.get(material, "yield_strength", T=20.0)
        strength_name = "fy"
    except Exception:
        strength = db.get(material, "compressive_strength", T=20.0)
        strength_name = "sigma_c"
    return {"material": material, "E": E, "nu": nu,
            "sigma_cr_Pa": sigma_cr, "strength_name": strength_name,
            "sigma_cr_over_strength": sigma_cr / strength}


def ice_flexural_load(db, h: float = 1.0, B: float = 10.0) -> dict:
    """ISO 19906 flexural-failure load scale: F ~ 0.21 sigma_f B h^2 / L_c.

    The characteristic length L_c = (E h^3 / (12 (1-nu^2) rho_w g))^(1/4) is
    evaluated with the database value of E, nu, rho of ice and water, i.e. the
    sample consumes *five* database records, not a single constant.
    """
    E = db.get("sea_ice", "E", T=-10.0)
    nu = db.get("sea_ice", "poisson_ratio", T=-10.0)
    sigma_f = db.get("sea_ice", "flexural_strength", T=-10.0)
    rho_ice = db.get("sea_ice", "density", T=-10.0)
    # NOTE: the v1 tables start at 0 degC (ITTC band); a freezing-point
    # seawater density would need the UNESCO EOS extension (see docs gap list)
    rho_w = db.get("sea_water", "density", T=0.0)
    g = 9.81
    L_c = (E * h ** 3 / (12.0 * (1.0 - nu ** 2) * rho_w * g)) ** 0.25
    F = 0.21 * sigma_f * B * h ** 2 / L_c
    return {"sigma_f_Pa": sigma_f, "L_c_m": L_c, "F_N": F,
            "rho_ice_over_rho_water": rho_ice / rho_w}


# --------------------------------------------------------------------------- #
def main() -> None:
    db = load_db()

    print("=== A. resistance (seawater, ITTC 7.5-02-01-03) ===")
    for key, value in resistance_sample(db).items():
        print(f"  {key:12s}: {value}")

    print("\n=== B. compressible DG initial state (air, M=0.5) ===")
    for key, value in compressible_state(db).items():
        print(f"  {key:12s}: {value}")

    print("\n=== C. SWE bed friction (medium sand) ===")
    for key, value in manning_bed_friction(db).items():
        print(f"  {key:12s}: {value}")

    print("\n=== D. structural / ice screening ===")
    for material in ("AH36", "EH40", "alu_5083_H116", "frp_glass_polyester"):
        res = plate_buckling(db, material)
        print(f"  {material:20s} sigma_cr = {res['sigma_cr_Pa'] / 1e6:6.2f} MPa"
              f"  (sigma_cr/{res['strength_name']} = "
              f"{res['sigma_cr_over_strength']:.3f})")
    ice = ice_flexural_load(db)
    print(f"  ice flexural load F = {ice['F_N'] / 1e6:.3f} MN"
          f"  (L_c = {ice['L_c_m']:.2f} m, sigma_f = "
          f"{ice['sigma_f_Pa'] / 1e6:.2f} MPa)")

    print("\n=== provenance sample ===")
    res = db.query("sea_water", "nu", T=15.0, unit="cSt")
    print(f"  seawater nu(15 degC) = {res.value:.4f} cSt")
    print(f"  source: {res.source['ref']}")


if __name__ == "__main__":
    main()