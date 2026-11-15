# Feature-level certification — CASE-SHU-OSHER round-2 reference

Anchor: `XVAL-DG-ROUND2-CERTIFIED-REFERENCE` · gate τ = 3.0 % · U/τ ≤ 0.333

Wave-train window: x ∈ [-2.7031, 2.2894] (analytic head +0.05, analytic shock −0.1), plateau ρ = 3.857143.

## 1. Functional certification (WENO5 / WENO7 ladders)

| feature | kind | value (WENO5) | U (%) | p | WENO7 value | Δ family (%) | Δ/(U5+U7) | G0 | G1 |
|---|---|---|---|---|---|---|---|---|---|
| amplitude | extremum | 0.827972 | 0.1169 | 1.6 | 0.827565 | 0.0492 | 0.32 | yes | yes |
| crest | extremum | 4.68041 | 2.616e-05 | 4.7 | — | — | — | NO | NO |
| rms | integral | 0.299924 | 0.07617 | 1.51 | — | — | — | NO | NO |
| l1 | integral | — | — | -0.715 | 0.233346 | — | — | NO | NO |
| peak_x | phase | — | — | -0.585 | — | — | — | NO | NO |

## 2. DG legs vs the certified features

| leg | profile L1 % (round 2) | amplitude err % | PASS/MARG/FAIL |
|---|---|---|---|
| DG P1 N=200 CFL=0.4 | 2.46 | 68.445 | 0/0/1 |
| DG P1 N=400 CFL=0.4 | 1.55 | 42.765 | 0/0/1 |
| DG P1 N=800 CFL=0.4 | 0.579 | 18.410 | 0/0/1 |
| DG P2 N=200 CFL=0.4 | 2.42 | 67.652 | 0/0/1 |
| DG P2 N=400 CFL=0.4 | 1.52 | 41.253 | 0/0/1 |
| DG P2 N=800 CFL=0.4 | 0.567 | 17.707 | 0/0/1 |
| DG P3 N=200 CFL=0.4 | 2.43 | 67.526 | 0/0/1 |
| DG P3 N=400 CFL=0.4 | 1.52 | 41.118 | 0/0/1 |
| DG P3 N=800 CFL=0.4 | 0.67 | 17.433 | 0/0/1 |
| DG P2 N=400 CFL=0.1 | 1.52 | 41.442 | 0/0/1 |
| DG P2 N=400 CFL=0.2 | 1.52 | 41.398 | 0/0/1 |

## 3. Supplementary — functionals certified by one family only (G0 refuses them; informational)

| feature | certified by | value | U (%) | p | worst DG err % | median % | FAIL |
|---|---|---|---|---|---|---|---|
| crest | weno5 | 4.68041 | 2.616e-05 | 4.7 | 11.73 | 6.986 | 8 |
| rms | weno5 | 0.299924 | 0.07617 | 1.51 | 53.6 | 34.66 | 11 |
| l1 | weno7 | 0.233346 | 0.367 | 0.848 | 47.64 | 28.93 | 11 |
| peak_x | None | — | — | — | — | — | None |

## 4. Same-grid cross-scheme comparison

| feature | N | WENO5 err % | WENO7 err % | DG P1 % | DG P2 % | DG P3 % |
|---|---|---|---|---|---|---|
| amplitude | 800 | 2.125 | 0.2651 | 18.41 | 17.71 | 17.43 |
| amplitude | 1600 | 0.8614 | 0.09811 | — | — | — |
| amplitude | 3200 | 0.2839 | 0.02851 | — | — | — |
| amplitude | 6400 | 0.09355 | 0.01964 | — | — | — |

## 5. Decisions

- **amplitude**: reference 0.827972, U = 0.1169 %; worst DG leg 68.445 %, median 41.398 %, FAIL 11/11, PASS_MARGINAL 0.

- **G4 profile gate sufficient: NO** (11 counterexample(s))
  - `DG P1 N=200 CFL=0.4`: profile L1 2.46 % PASS, but amplitude 68.445 % FAIL
  - `DG P1 N=400 CFL=0.4`: profile L1 1.55 % PASS, but amplitude 42.765 % FAIL
  - `DG P1 N=800 CFL=0.4`: profile L1 0.579 % PASS, but amplitude 18.410 % FAIL
  - `DG P2 N=200 CFL=0.4`: profile L1 2.42 % PASS, but amplitude 67.652 % FAIL
  - `DG P2 N=400 CFL=0.4`: profile L1 1.52 % PASS, but amplitude 41.253 % FAIL
  - `DG P2 N=800 CFL=0.4`: profile L1 0.567 % PASS, but amplitude 17.707 % FAIL
  - `DG P3 N=200 CFL=0.4`: profile L1 2.43 % PASS, but amplitude 67.526 % FAIL
  - `DG P3 N=400 CFL=0.4`: profile L1 1.52 % PASS, but amplitude 41.118 % FAIL
  - `DG P3 N=800 CFL=0.4`: profile L1 0.67 % PASS, but amplitude 17.433 % FAIL
  - `DG P2 N=400 CFL=0.1`: profile L1 1.52 % PASS, but amplitude 41.442 % FAIL
  - `DG P2 N=400 CFL=0.2`: profile L1 1.52 % PASS, but amplitude 41.398 % FAIL
