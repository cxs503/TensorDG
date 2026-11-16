"""Independent mean-field reference for a **two-component Shan-Chen (SCMP)
planar interface** -- derived from thermodynamics only, no LBM code involved.

Why this file exists
--------------------
``tensorlbm.mcmp`` ships a two-component Shan-Chen solver, but its two-phase
capability has never been checked against a reference that was computed
independently of the solver (the in-repo ``static-droplet-laplace`` row of
``t3_matrix`` is WITHHELD at 38 % dispersion, and ``setup_laplace_test`` uses a
hand-waved surface-tension estimate).  A *planar* interface is the cleanest
configuration in which an exact reference exists, so this module supplies it.

Model and sign conventions (as implemented in ``tensorlbm.cross_force``)
-----------------------------------------------------------------------
The implemented inter-component force is

    F_a = + G_ab * psi_a(x) * sum_i w_i psi_b(x + e_i) e_i
        ->  G_ab * cs^2 * psi_a * grad psi_b          (isotropic leading order)

so the force has a **plus** sign in front of ``G`` (note: the docstring of
``cross_force.compute_force`` still writes a minus sign -- see the audit note in
``docs/XVAL-LBM-SCMP-TWO-COMPONENT-PLANAR-INTERFACE.md``).  Two consequences
follow from that convention and are used below:

* the mixture mechanical pressure is
      p = cs^2 (rho_1 + rho_2 - G_12 psi_1 psi_2)                       (1)
  (total force = +G cs^2 grad(psi_1 psi_2), i.e. a pure gradient, so it is
  absorbed into the pressure with weight 1 -- there is no 1/2 here because the
  two interaction partners are distinct species; the 1/2 of the single-component
  model comes from double counting a self-interaction);
* demixing (two-phase separation) requires ``G_12 < 0``; ``G_12 > 0`` mixes.

For the **linear pseudopotential** ``psi(rho) = rho`` (``pseudopotential_rho``)
the model is thermodynamically consistent and an exact Helmholtz free energy
density exists,

    f/(cs^2) = rho_1 ln rho_1 - rho_1 + rho_2 ln rho_2 - rho_2 - G_12 rho_1 rho_2   (2)

whose derivatives reproduce (1).  The chemical potentials are then

    mu_1 = cs^2 (ln rho_1 - G_12 rho_2),   mu_2 = cs^2 (ln rho_2 - G_12 rho_1)      (3)

and the bulk tie-line of a symmetric binary mixture with equal component masses
(``<rho_1> = <rho_2> = rho_bar``, so ``rho_hi + rho_lo = 2 rho_bar``) follows
from ``mu_1(A) = mu_1(B)``:

    ln(rho_hi/rho_lo) = -G_12 (rho_hi - rho_lo) = |G_12| (rho_hi - rho_lo)          (4)

The linearised stability of the symmetric mixture gives the **critical coupling**

    |G_c| = 1/rho_bar        (psi = rho)                                            (5)

For a non-linear pseudopotential suchas ``psi = 1 - exp(-rho)`` equations (2)-(4)
do **not** exist: the SC "equation of state" (1) with nonlinear psi is not
integrable in two components (the Gibbs-Duhem cross-derivative identity fails),
so no exact tie-line reference can be constructed for it.  For that case this
module deliberately returns ``WITHHELD`` instead of an invented number; only the
mechanical-pressure invariant (1) and the linearised criterion
``|G| psi'(rho_bar)^2 > 1/rho_bar`` are used.

Usage
-----
    from scmp_planar_reference import coexistence_tieline, mechanical_pressure
    rhi, rlo = coexistence_tieline(G=-3.0, rho_bar=0.5)   # linear psi only

Command line self-test (cross-checks the two independent numerical routes):

    python TensorLBM/tools/scmp_planar_reference.py --selftest
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

CS2 = 1.0 / 3.0


# ---------------------------------------------------------------------------
# Thermodynamics of the two-component SCMP model
# ---------------------------------------------------------------------------

def pseudopotential(rho: float, kind: str = "linear") -> float:
    """Pseudopotential used by the benchmarked arms."""
    if kind == "linear":
        return rho
    if kind == "exp":
        return 1.0 - math.exp(-rho)
    raise ValueError(f"unknown pseudopotential kind: {kind!r}")


def pseudopotential_derivative(rho: float, kind: str = "linear") -> float:
    if kind == "linear":
        return 1.0
    if kind == "exp":
        return math.exp(-rho)
    raise ValueError(f"unknown pseudopotential kind: {kind!r}")


def mechanical_pressure(rho1: float, rho2: float, G: float,
                        psi_kind: str = "linear", cs2: float = CS2) -> float:
    """Mixture mechanical pressure, equation (1).

    ``p = cs^2 (rho_1 + rho_2 - G psi_1 psi_2)`` for the implemented force sign
    (``F = +G psi grad psi``).
    """
    return cs2 * (rho1 + rho2
                  - G * pseudopotential(rho1, psi_kind) * pseudopotential(rho2, psi_kind))


def free_energy_density(rho1: float, rho2: float, G: float,
                        cs2: float = CS2) -> float:
    """Ideal + interaction Helmholtz free energy density, equation (2).

    Only valid for the linear pseudopotential (see module docstring).
    """
    if rho1 <= 0.0 or rho2 <= 0.0:
        return math.inf
    return cs2 * (rho1 * math.log(rho1) - rho1
                  + rho2 * math.log(rho2) - rho2
                  - G * rho1 * rho2)


def chemical_potentials(rho1: float, rho2: float, G: float,
                        cs2: float = CS2) -> Tuple[float, float]:
    """mu_1, mu_2 from equation (3) (linear pseudopotential only)."""
    return (cs2 * (math.log(rho1) - G * rho2),
            cs2 * (math.log(rho2) - G * rho1))


def critical_coupling(rho_bar: float = 0.5, psi_kind: str = "linear") -> float:
    """|G_c| above which the symmetric mixture is unstable against demixing.

    ``|G_c| = 1 / (rho_bar * psi'(rho_bar)^2)``.
    """
    dpsi = pseudopotential_derivative(rho_bar, psi_kind)
    return 1.0 / (rho_bar * dpsi * dpsi)


# ---------------------------------------------------------------------------
# Route A: root of the transcendental coexistence condition (4)
# ---------------------------------------------------------------------------

def coexistence_rho_hi(A: float, rho_bar: float = 0.5,
                       tol: float = 1e-15, max_iter: int = 400) -> float:
    """Solve ``ln(x/(1-x)) = A (2x-1)`` for ``x = rho_hi``, ``A = |G|``.

    ``rho_lo = 2 rho_bar - rho_hi``; ``A = |G|``.  Bisection on ``(rho_bar, 1)``;
    the left-hand side is strictly increasing in ``x``, the right-hand side is
    increasing too, so the difference has a single sign change on the branch we
    want (the trivial ``x = rho_bar`` root is excluded by starting above it).
    """
    if A <= 1.0 / rho_bar:
        # below (or at) the critical coupling the only solution is the mixture
        return rho_bar

    def f(x: float) -> float:
        return math.log(x / (1.0 - x)) - A * (2.0 * x - 1.0) * (rho_bar / 0.5)

    lo = rho_bar + 1e-14
    hi = 1.0 - 1e-14
    if f(hi) < 0.0:
        return rho_bar
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Route B: direct minimisation of the two-phase free energy (independent of A)
# ---------------------------------------------------------------------------

def coexistence_rho_hi_by_energy(G: float, rho_bar: float = 0.5,
                                 n_grid: int = 200001, n_golden: int = 200) -> float:
    """Minimise the two-phase free energy per unit area over the split point.

    The material is split into half A-rich and half B-rich, so with equal
    component masses ``rho_lo = 2 rho_bar - rho_hi``.  The functional minimised
    is

        F(x) = f(x, 2*rho_bar - x) + f(2*rho_bar - x, x)

    which is the free energy per unit volume of the fully phase-separated state
    (interfacial terms are positive and do not move the bulk split).  No use is
    made of equation (4), so this is an independent route.
    """
    if abs(G) <= 1.0 / rho_bar:
        return rho_bar

    def F(x: float) -> float:
        y = 2.0 * rho_bar - x
        if x <= 0.0 or y <= 0.0:
            return math.inf
        return (free_energy_density(x, y, G) + free_energy_density(y, x, G))

    lo, hi = rho_bar + 1e-9, 2.0 * rho_bar - 1e-9
    grid = [lo + (hi - lo) * i / (n_grid - 1) for i in range(n_grid)]
    vals = [F(x) for x in grid]
    i = min(range(n_grid), key=lambda k: vals[k])
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, n_grid - 1)]
    # golden-section refinement inside the bracketing interval
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    for _ in range(n_golden):
        if F(c) < F(d):
            b = d
        else:
            a = c
        c = b - gr * (b - a)
        d = a + gr * (b - a)
    return 0.5 * (a + b)


def coexistence_tieline(G: float, rho_bar: float = 0.5,
                        psi_kind: str = "linear") -> Tuple[float, float]:
    """Bulk densities ``(rho_hi, rho_lo)`` of the planar coexistence state.

    Raises ``NotImplementedError`` for non-linear pseudopotentials, whose
    two-component SC "EOS" admits no exact free energy (see module docstring):
    the honest answer there is WITHHELD, not a number.
    """
    if psi_kind != "linear":
        raise NotImplementedError(
            "no exact tie-line reference exists for a non-linear pseudopotential "
            "(the two-component Shan-Chen EOS is not integrable); report WITHHELD"
        )
    if G >= 0.0:
        # repulsive-in-the-implemented-sign coupling mixes the species
        return rho_bar, rho_bar
    rhi = coexistence_rho_hi(abs(G), rho_bar)
    return rhi, 2.0 * rho_bar - rhi


# ---------------------------------------------------------------------------
# Derived predictions used by the benchmark
# ---------------------------------------------------------------------------

def interface_density_dip(G: float, rho_bar: float = 0.5,
                          psi_kind: str = "linear",
                          cs2: float = CS2) -> float:
    """Total density at the centre of the interface, from constant pressure (1).

    At the centre of a symmetric interface ``rho_1 = rho_2 = s/2``; requiring the
    mechanical pressure to equal the bulk value gives

        s + |G| psi(s/2)^2 = 2 rho_bar + |G| psi(rho_hi) psi(rho_lo)

    which is solved by bisection on ``s``.  This is the Shan-Chen "density dip":
    the interface must be *less* dense than either bulk phase so that (1) stays
    flat.  For ``|G|`` just above critical the dip is small; it grows with the
    coupling, and a large dip is the classical stiffness of SC two-phase runs.
    """
    p_bulk = mechanical_pressure(rho_bar, rho_bar, G, psi_kind, cs2) if G >= 0 else None
    rhi, rlo = coexistence_tieline(G, rho_bar, psi_kind) if psi_kind == "linear" else (rho_bar, rho_bar)
    p_bulk = mechanical_pressure(rhi, rlo, G, psi_kind, cs2)

    def f(s: float) -> float:
        return mechanical_pressure(0.5 * s, 0.5 * s, G, psi_kind, cs2) - p_bulk

    lo, hi = 1e-9, 10.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def profile_pressure_flatness(rho1, rho2, G: float, psi_kind: str = "linear",
                              cs2: float = CS2) -> Dict[str, float]:
    """Exact invariant of a resting planar interface: (1) must be constant in y.

    ``rho1``/``rho2`` are 1-D (per-row) density profiles.  Returns the mean
    pressure and the peak-to-peak spread normalised by the mean -- a quantity
    that is *zero* for a converged equilibrium state and that grows with any
    force-scaling error (a factor-2 error in ``G``, a wrong sign, a missing
    pressure weight all show up here as a large spread).
    """
    n = len(rho1)
    p = [mechanical_pressure(rho1[i], rho2[i], G, psi_kind, cs2) for i in range(n)]
    mean = sum(p) / n
    spread = max(p) - min(p)
    return {
        "pressure_mean": mean,
        "pressure_min": min(p),
        "pressure_max": max(p),
        "pressure_spread": spread,
        "pressure_spread_relative": spread / abs(mean) if mean else math.inf,
    }


def tie_line_relative_error(rho_hi_sim: float, rho_lo_sim: float, G: float,
                            rho_bar: float = 0.5,
                            psi_kind: str = "linear") -> Dict[str, float]:
    """Relative error of the measured bulk densities against the reference."""
    rhi, rlo = coexistence_tieline(G, rho_bar, psi_kind)
    return {
        "rho_hi_reference": rhi,
        "rho_lo_reference": rlo,
        "err_rho_hi": abs(rho_hi_sim - rhi) / rhi,
        "err_rho_lo": abs(rho_lo_sim - rlo) / rlo,
        "err_max": max(abs(rho_hi_sim - rhi) / rhi, abs(rho_lo_sim - rlo) / rlo),
        "amplitude_sim": rho_hi_sim - rho_lo_sim,
        "amplitude_reference": rhi - rlo,
    }


# ---------------------------------------------------------------------------
# Self-test: two independent routes must agree
# ---------------------------------------------------------------------------

def selftest(verbose: bool = True) -> Dict[str, float]:
    out: Dict[str, float] = {}

    # 1. below critical coupling the mixture is stable -> degenerate tie-line
    assert coexistence_tieline(-1.5) == (0.5, 0.5)
    assert coexistence_tieline(+3.0) == (0.5, 0.5)

    # 2. critical coupling is where the transcendental loses its non-trivial root
    gc = critical_coupling(0.5)
    out["critical_coupling"] = gc
    assert abs(gc - 2.0) < 1e-12, gc

    # 3. routes A and B (transcendental vs. direct free-energy minimisation)
    #    must agree; they share no algebra
    worst = 0.0
    for A in (2.05, 2.2, 2.5, 3.0, 4.0, 6.0):
        a = coexistence_rho_hi(A)
        b = coexistence_rho_hi_by_energy(-A, n_grid=40001, n_golden=120)
        worst = max(worst, abs(a - b))
        if verbose:
            print(f"  |G|={A:5.2f}  route A rho_hi={a:.12f}  route B rho_hi={b:.12f}  "
                  f"|diff|={abs(a - b):.2e}")
    out["route_agreement"] = worst
    assert worst < 1e-6, worst

    # 4. at the tie-line the chemical potential and the mechanical pressure
    #    of the two phases coincide (both are implied by the construction)
    G = -3.0
    rhi, rlo = coexistence_tieline(G)
    mu_a = chemical_potentials(rhi, rlo, G)
    mu_b = chemical_potentials(rlo, rhi, G)
    out["mu_imbalance"] = max(abs(mu_a[0] - mu_b[0]), abs(mu_a[1] - mu_b[1]))
    p_a = mechanical_pressure(rhi, rlo, G)
    p_b = mechanical_pressure(rlo, rhi, G)
    out["pressure_imbalance"] = abs(p_a - p_b)
    assert out["mu_imbalance"] < 1e-12, out["mu_imbalance"]
    assert out["pressure_imbalance"] < 1e-12, out["pressure_imbalance"]

    # 5. mass consistency of the construction: rho_hi + rho_lo = 2 rho_bar
    assert abs(rhi + rlo - 1.0) < 1e-12

    # 6. the interface must be less dense than the bulks (density dip > 0)
    dip = interface_density_dip(G)
    out["interface_total_density"] = dip
    out["dip_relative_to_bulk"] = ((rhi + rlo) - dip) / (rhi + rlo)
    assert dip < rhi + rlo, (dip, rhi, rlo)

    # 7. an exactly flat pressure profile reports zero spread
    flat = profile_pressure_flatness([rhi] * 5 + [rlo] * 5, [rlo] * 5 + [rhi] * 5, G)
    out["flat_profile_spread"] = flat["pressure_spread_relative"]
    assert flat["pressure_spread_relative"] < 1e-14

    if verbose:
        for k, v in out.items():
            print(f"  {k:28s} = {v:.6e}")
    return out


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        print("scmp_planar_reference self-test")
        selftest()
        print("OK")
    else:
        for G in (-2.5, -3.0, -4.0):
            rhi, rlo = coexistence_tieline(G)
            print(f"G={G:+.2f}  tie-line=({rhi:.6f}, {rlo:.6f})  "
                  f"p_bulk={mechanical_pressure(rhi, rlo, G):.6f}  "
                  f"s_interface={interface_density_dip(G):.6f}")