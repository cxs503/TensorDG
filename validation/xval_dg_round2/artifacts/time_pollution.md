# XVAL-DG-ROUND2-CERTIFIED-REFERENCE — time-discretisation pollution audit

**Verdict: `TIME_CLEAN`** (accepted = True) — question: *is the published certified reference polluted by the time step it was generated at?*

CFL series [0.3, 0.15, 0.075] (ratio 2, fixed grids), reference grid N = 3200, band = Roache GCI with Fs = 1.25.

| criterion | statement | measured | pass |
|---|---|---|---|
| T0 | temporal successive differences decrease | {800: True, 1600: True, 3200: True, 6400: True} | True |
| T1 | delta_t / d_fine <= 0.1 on N=3200 | 0.046439 (CLEAN) | True |
| T2 | temporal order >= 0.5 | 1.9999999999998446 | True |
| T3 | band shift <= 0.25 and corrected ladder CERTIFIED | 0.012272 | True |
| T4 | no 3 % decision flip, leg move <= 0.5 pp | 0 flips, max move 0.04137588557648986 pp | True |

## Time leg per grid

| N | delta_t = rel L1(0.3, 0.15) % | rel L1(0.15, 0.075) % | p_t | d_fine (spatial) % | R = delta_t/d_fine |
|---|---|---|---|---|---|
| 800 | 4.138878e-01 | 1.034719e-01 | 2.0000 | 3.888837e+00 | 0.1064 |
| 1600 | 1.045970e-01 | 2.614924e-02 | 2.0000 | 1.456051e+00 | 0.0718 |
| 3200 | 2.627230e-02 | 6.568075e-03 | 2.0000 | 5.657319e-01 | 0.0464 |
| 6400 | 6.581477e-03 | 1.645369e-03 | 2.0000 | 2.281861e-01 | 0.0288 |

## Reference: raw (CFL = 0.3) vs time-corrected (CFL -> 0)

| ladder | verdict | observed order | band % | reference grid |
|---|---|---|---|---|
| raw three-level (published round 2) | CERTIFIED | 1.376846 | 0.445327 | 1600 |
| raw four-level (current shipped) | CERTIFIED | 1.314797 | 0.192169 | 3200 |
| time-corrected three-level | CERTIFIED | 1.193321 | 0.450793 | 1600 |

## DG legs: does the reference choice move a published verdict?

| configuration | err % (raw ref) | err % (time-corrected ref) | decision 3% (raw) | decision 3% (corrected) |
|---|---|---|---|---|
| DG P1 N=200 CFL=0.4 | 4.961733 | 5.003108 | FAIL | FAIL |
| DG P1 N=400 CFL=0.4 | 2.139138 | 2.179401 | PASS | PASS |
| DG P1 N=800 CFL=0.4 | 0.910532 | 0.950311 | PASS | PASS |
| DG P2 N=200 CFL=0.4 | 4.961733 | 5.003108 | FAIL | FAIL |
| DG P2 N=400 CFL=0.4 | 2.139138 | 2.179401 | PASS | PASS |
| DG P2 N=800 CFL=0.4 | 0.910532 | 0.950311 | PASS | PASS |
| DG P3 N=200 CFL=0.4 | 4.961733 | 5.003108 | FAIL | FAIL |
| DG P3 N=400 CFL=0.4 | 2.139138 | 2.179401 | PASS | PASS |
| DG P3 N=800 CFL=0.4 | 0.910532 | 0.950311 | PASS | PASS |
| DG P2 N=400 CFL=0.1 | 2.139138 | 2.179401 | PASS | PASS |
| DG P2 N=400 CFL=0.2 | 2.139138 | 2.179401 | PASS | PASS |

## Thresholds (predeclared)

* `POLLUTION_MAX = 0.1` (CLEAN), `POLLUTION_MARGINAL = 0.25` (MARGINAL ceiling)
* `P_MIN_T = 0.5`, `BAND_SHIFT_MAX = 0.25`, `LEG_STABILITY_MAX_PP = 0.5`

