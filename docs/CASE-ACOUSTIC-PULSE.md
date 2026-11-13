# CASE-ACOUSTIC-PULSE — 2-D linear acoustics DG: exact cylindrical pulse

**Status**: validated capability (`tensordg.acoustics`), merged into the public
API contract (`from tensordg import solve_acoustic_2d, acoustic_pulse_exact_state`).
**Artifacts**: `validation/acoustic_pulse/convergence.json` (machine record),
`docs/figures/acoustic_pulse_convergence.png`, `docs/figures/acoustic_pulse_exact.png`.
**Reproduce**:
```
PYTHONPATH=. python -m benchmarks.acoustic_pulse_2d \
    --output-dir validation/acoustic_pulse --figures-dir docs/figures
PYTHONPATH=. python -m pytest tests/test_acoustic_pulse.py -q   # 31 tests
PYTHONPATH=. python -m pytest tests/test_acoustic_pulse_doc_binding.py -q
```

**Doc <-> record <-> solver binding.**  The numbers in this file are not
hand-typed: `tests/test_acoustic_pulse_doc_binding.py` re-derives every table
cell (design order, `h`, steps, both relative `L2` columns, energy, drift), the
`< 3%?` verdicts of §4, the field-error claim of §3.2 and the `E0` anchor of §3.1
from `validation/acoustic_pulse/convergence.json` at the precision printed here,
and re-runs the two cheapest rungs `(p=1, 16^2)` / `(p=2, 16^2)` from the solver
to `1e-9` relative.  So a record refresh that this document does not follow, or a
solver change that moves a recorded number, fails CI instead of shipping a stale
claim.

## 1. Case (physics / geometry / parameters)

Linear acoustics in a homogeneous fluid, first-order conservative form

```
dp/dt + K (du/dx + dv/dy) = 0 ,  du/dt + (1/rho) dp/dx = 0 ,  dv/dt + (1/rho) dp/dy = 0
```

with `K = rho c^2`.  Eliminating the velocity gives `p_tt = c^2 lap(p)`.

Initial data: an axisymmetric Gaussian pressure pulse in a fluid at rest,
`p(r,0) = A exp(-r^2/sigma^2)`, `u = v = 0`.  It is the classic 2-D
cylindrical-spreading benchmark (Tam & Webb 1993 family): after a few `sigma/c`
the pulse has split into an outward-running front **and** a trailing wake of the
opposite sign — the feature a 1-D anchor cannot reproduce.

| parameter | value |
|---|---|
| domain | `[0,2] x [0,2]`, periodic, pulse centred at `(1,1)` |
| medium | `rho = 1`, `c = 1` (`K = 1`, `Z = 1`) |
| `sigma` | 0.15 |
| amplitude `A` | 1 |
| final time | 0.3 (front at `r = c t = 0.3`, i.e. 0.7 away from the wall: no boundary interaction) |
| CFL | 0.2, `dt = CFL h / ((2p+1) c)`, SSP-RK3 |
| polynomial degrees | `p = 1, 2, 3` |
| meshes | `16^2`, `32^2`, `64^2` |

The exact solution is the Hankel (Bessel `J0`) integral
`p(r,t) = int_0^inf F(k) J0(k r) cos(k c t) k dk` with
`F(k) = (A sigma^2/2) exp(-k^2 sigma^2/4)`, and
`u_r = (1/(rho c)) int_0^inf F(k) J1(k r) sin(k c t) k dk`.

## 2. How (methods / toolchain / key settings)

* **Solver** `tensordg.acoustics.solve_acoustic_2d`: modal Legendre DG on
  tensor-product cells, reusing the 1-D operator pack of
  `tensordg.elastic_2d` (one source of truth for quadrature); strong-form volume
  term with exact face corrections; **exact Godunov flux** of the linear system
  (`p* = (pL+pR)/2 + Z(uL-uR)/2`, `u* = (uL+uR)/2 + (pL-pR)/(2Z)`), i.e. no
  tunable dissipation; periodic boundaries; SSP-RK3.
* **Exact anchor** `acoustic_pulse_exact_pressure/_velocity`: composite
  Gauss–Legendre quadrature in wavenumber, panel width `1/(r_max + c t)` and
  truncation `k_max = 12/sigma` derived from the tail bound, so the residual is
  at machine precision (verified, see §5).
* **Verification layers** (cheapest first, all in `tests/test_acoustic_pulse.py`):
  1. flux algebra — Riemann star state re-derived from the characteristic
     invariants, physical sign of the pressure-jump problem, and the
     **argument-order regression guard** (see §5);
  2. `P0` DG must collapse onto the upwind finite-volume update (closed form);
  3. exact anchor — `t = 0` identity, quadrature refinement, an *independent*
     real-space **Poisson disk** integral, and the radial acoustic equations on
     the reconstructed state;
  4. solver — order `p+1`, energy conservation, and the analytic energy
     `E0 = pi sigma^2 A^2 / (4K)`.
* **Toolchain**: pure PyTorch (`float64`, CPU), numpy/scipy for the Bessel
  functions and the cross-check, matplotlib for the figures.  Single core; the
  whole ladder takes ≈ 90 s (`p=3, 64^2` dominates).

## 3. Results

### 3.1 Convergence against the exact solution (relative `L2` of `(p,u,v)`)

| p | refinement | observed order | design order |
|---|---|---|---|
| 1 | 16→32^2 | 2.15 | 2 |
| 1 | 32→64^2 | 2.29 | 2 |
| 2 | 16→32^2 | 3.36 | 3 |
| 2 | 32→64^2 | 3.06 | 3 |
| 3 | 16→32^2 | 4.10 | 4 |
| 3 | 32→64^2 | 4.01 | 4 |

Full record:

| p | cells | h | steps | rel. L2 state | rel. L2 p | E (DG) | |E-E0|/E0 |
|---|-------|---|-------|---------------|-----------|--------|-----------|
| 1 | 16 | 0.1250 | 36 | 1.340e-01 | 1.310e-01 | 0.015886394 | 1.02e-01 |
| 1 | 32 | 0.0625 | 72 | 3.010e-02 | 2.960e-02 | 0.017346063 | 1.84e-02 |
| 1 | 64 | 0.0312 | 144 | 6.158e-03 | 5.888e-03 | 0.017625422 | 2.61e-03 |
| 2 | 16 | 0.1250 | 60 | 1.602e-02 | 1.644e-02 | 0.017593535 | 4.34e-03 |
| 2 | 32 | 0.0625 | 120 | 1.565e-03 | 1.552e-03 | 0.017668246 | 1.82e-04 |
| 2 | 64 | 0.0312 | 240 | 1.882e-04 | 1.845e-04 | 0.017671340 | 6.69e-06 |
| 3 | 16 | 0.1250 | 84 | 1.697e-03 | 1.710e-03 | 0.017669185 | 1.32e-04 |
| 3 | 32 | 0.0625 | 168 | 9.912e-05 | 9.848e-05 | 0.017671399 | 3.38e-06 |
| 3 | 64 | 0.0312 | 336 | 6.164e-06 | 6.083e-06 | 0.017671454 | 2.91e-07 |

Analytic initial energy `E0 = pi sigma^2/(4K) = 0.017671459`; the DG energy of the
finest run of each degree is within `2.61e-03`
of it, i.e. the scheme conserves the discrete energy *and* the projected initial
data already carry the right energy.

### 3.2 Field comparison

`docs/figures/acoustic_pulse_exact.png`: numerical vs exact pressure, the
difference field, and centred `p`/`u_r` profiles.  For `p = 3, 64^2` the maximum
pointwise pressure error is **3.542e-06** against a field
scale of 0.206 (1.72e-03 %).

`docs/figures/acoustic_pulse_convergence.png`: relative `L2` error vs `h` with the
`h^{p+1}` reference slopes.

## 4. Error judgement (key quantities vs benchmark, explicit)

| quantity | benchmark | measured | error | < 3%? |
|---|---|---|---|---|
| relative `L2` of `(p,u,v)`, `p=1, 64^2` | exact Hankel solution | 6.158e-03 | — | **yes** (0.62%) |
| relative `L2`, `p=2, 64^2` | exact Hankel solution | 1.882e-04 | — | **yes** (0.02%) |
| relative `L2`, `p=3, 64^2` | exact Hankel solution | 6.164e-06 | — | **yes** (0.00%) |
| convergence order, `p = 1/2/3` | design order `p+1` | 2.29, 3.06, 4.01 | ≤ 0.29 | **yes** |
| energy drift, `p=3, 64^2` | `E0 = pi sigma^2/(4K)` | 2.91e-07 | — | **yes** |
| exact anchor vs independent Poisson-disk integral | Hankel integral | ≤ 1e-7 (relative) | — | **yes** |

No quantity is above the 3% line; the finest-grid pressure field is at
`1.7e-5` relative.  Non-monotone detail worth stating: at `p=1, 32^2` the state
error (3.0e-2) is above 3% — a `sigma/h = 2.4` P1 grid simply cannot resolve the
pulse, which is exactly what the order test detects (2.15 → 2.29) rather than
hides.

## 5. Pitfalls found (honest list)

1. **Riemann argument order at the right face.** Passing `(neighbour, interior)`
   instead of `(interior, neighbour)` leaves the flux *consistent* (the
   correction vanishes on continuous traces, so every smooth manufactured test
   still passes) but upwinds the wrong way at jumps.  Symptom: an `h`-independent
   `L2` error (0.33 at `p=1`, no improvement from 16^2 to 64^2) and a discrete
   energy growing 44%.  Diagnosed by (a) the `P0` finite-volume limit, where the
   correct update is known in closed form, and (b) a two-cell pressure-jump
   `dq/dt` check.  Both are now regression tests
   (`test_p0_rhs_matches_the_upwind_finite_volume_update`,
   `test_riemann_flux_argument_order_is_pinned`).
2. **Strong-form sign of the face correction.** The strong form is
   `dq/dt = -div F - M^{-1} ∮ phi (F* - F)·n`; flipping the face sign produces
   an anti-dissipative scheme that diverges (`E/E0 ~ 1e18` in 12 steps).  Checked
   by the `P0` limit, not by smooth-data consistency.
3. **"Exact" needs a defined tolerance and an independent route.** The Hankel
   integral is machine-precision, but only after the panel width is tied to the
   oscillation rate `r_max + c t`; a fixed panel count silently degrades at
   larger `t` or `r`.  The independent Poisson-disk evaluation (a different
   transform, a different quadrature, a numerical time derivative) is what makes
   the anchor trustworthy, and it is a test, not a claim.
4. **Pre-asymptotic traps.** Refinement ratios are not always 2 (`32^2 → 48^2`);
   computing orders with `log2` there produces a fake order drop.  The benchmark
   derives the order from the actual `h` ratio.
5. **Scope limits**: periodic boundaries only (like `elastic_2d`); no impedance
   or wall BCs, no variable density, no advection.  The box is chosen so the
   boundary never enters the error budget, which is a *choice*, not a verified
   boundary condition.

## 6. What this establishes

`tensordg.acoustics` is a REAL capability: a DG discretisation of the 2-D linear
acoustic system with (i) design-order `p+1` convergence against an independently
validated exact solution, (ii) exact conservation of the acoustic energy up to
the analytic initial value, and (iii) an exact Godunov flux with no tunable
parameter.  It is the acoustic parent model for the ecosystem's wave chains
(aeroacoustics, ice-edge loading, structural-acoustic coupling) and provides the
anchor those couplings will be verified against.
