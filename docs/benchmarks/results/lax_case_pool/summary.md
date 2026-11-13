# CASE-LAX-DG -- measured results (generated, do not edit)

Case: Lax tube, gamma=1.4, t=0.13, x0=0.5, CFL=0.35

## analytic wave pattern

| quantity | value |
|---|---|
| pressure_star | 2.466098 |
| velocity_star | 1.528723 |
| density_star_left | 0.344568 |
| density_star_right | 1.304085 |
| x(rarefaction_head) | 0.157637 |
| x(rarefaction_tail) | 0.287229 |
| x(contact) | 0.698734 |
| x(shock) | 0.822312 |

## resolution ladder

| P | N | steps | rel L1 rho | rel L1 u | rel L1 p | analytic-region rel L1 rho | wall [s] |
|---|---|---|---|---|---|---|---|
| 1 | 100 | 555 | 2.32905e-02 | 1.03142e-02 | 5.22656e-03 | 4.46635e-03 | 14.1 |
| 1 | 200 | 1103 | 1.36030e-02 | 4.39831e-03 | 2.50089e-03 | 8.47904e-04 | 27.8 |
| 1 | 400 | 2199 | 7.21833e-03 | 1.95720e-03 | 1.10455e-03 | 1.30826e-04 | 56.4 |
| 2 | 100 | 913 | 1.61857e-02 | 1.14336e-02 | 6.36031e-03 | 4.46627e-03 | 22.0 |
| 2 | 200 | 1828 | 9.22761e-03 | 4.72106e-03 | 2.88266e-03 | 1.94404e-03 | 47.6 |
| 2 | 400 | 3662 | 4.52816e-03 | 1.66311e-03 | 1.09847e-03 | 6.12389e-04 | 97.2 |
| 3 | 100 | 1271 | 1.58470e-02 | 2.05087e-02 | 1.15437e-02 | 8.48553e-03 | 30.7 |
| 3 | 200 | 2547 | 1.21475e-02 | 1.69899e-02 | 9.56428e-03 | 7.34051e-03 | 59.1 |
| 3 | 400 | 5109 | 8.68646e-03 | 1.23092e-02 | 7.31418e-03 | 6.07272e-03 | 142.6 |

## gates

| gate | verdict | detail |
|---|---|---|
| C1 | PASS | rarefaction_head: 0.106 %; rarefaction_tail: 0.254 %; contact: 0.035 %; shock: 0.049 % |
| C2 | PASS | star_left.density: 0.006 %; star_left.velocity: 0.011 %; star_left.pressure: 0.003 %; star_right.density: 0.008 % |
| C3 | FAIL | p1.density: 0.914; p1.velocity: 1.168; p1.pressure: 1.179; p2.density: 1.027; p2.velocity: 1.505; p2.pressure: 1.392; p3.density: 0.484; p3.velocity: 0.465; p3.pressure: 0.387 |
| R3 | FAIL | p1.density: 2.696; p1.velocity: 1.567; p1.pressure: 1.622; p1.requirement: 1.700; p1.density_ratio: 1.586; p2.density: 1.667; p2.velocity: 2.112; p2.pressure: 1.896; p2.requirement: 2.550; p2.density_ratio: 0.654; p3.density: 0.274; p3.velocity: 0.361; p3.pressure: 0.337; p3.requirement: 3.400; p3.density_ratio: 0.080 |
| R5 | PASS | u=1.0.p1: order 2.008 (dev 0.41 %); u=2.0.p1: order 2.017 (dev 0.85 %); u=1.0.p2: order 3.001 (dev 0.03 %); u=2.0.p2: order 3.001 (dev 0.03 %) |
| TIME | PASS | p1.N100: worst share 0.25 %; p2.N200: worst share 1.19 %; p3.N200: worst share 1.90 % |

## temporal refinement (mesh fixed, dt -> dt/2 -> dt/4)

| leg | cfl | dt | steps | rel L1 rho | rel L1 u | temporal order (rho) | temporal share (rho) |
|---|---|---|---|---|---|---|---|
| p1 N=100 | 0.3500 | 1.4796e-05 | 555 | 2.32905e-02 | 1.03142e-02 | n/a | n/a |
| p1 N=100 | 0.1750 | 2.7064e-05 | 1109 | 2.32905e-02 | 1.03431e-02 | n/a | n/a |
| p1 N=100 | 0.0875 | 3.5394e-05 | 2217 | 2.32915e-02 | 1.03402e-02 | n/a | n/a |
| p2 N=200 | 0.3500 | 3.1684e-05 | 1828 | 9.22761e-03 | 4.72106e-03 | n/a | n/a |
| p2 N=200 | 0.1750 | 4.5187e-06 | 3656 | 9.25786e-03 | 4.75430e-03 | n/a | n/a |
| p2 N=200 | 0.0875 | 6.1469e-06 | 7311 | 9.26869e-03 | 4.77800e-03 | n/a | n/a |
| p3 N=200 | 0.3500 | 4.7994e-05 | 2547 | 1.21475e-02 | 1.69899e-02 | n/a | n/a |
| p3 N=200 | 0.1750 | 8.8522e-07 | 5094 | 1.22830e-02 | 1.72056e-02 | n/a | n/a |
| p3 N=200 | 0.0875 | 7.2573e-06 | 10184 | 1.23322e-02 | 1.72808e-02 | n/a | n/a |

## R3 region-rule nesting (retained measure per leg)

| degree | nested spread | legacy spread | legacy grows with refinement |
|---|---|---|---|
| p1 | 0.00e+00 | 2.40e-01 | True |
| p2 | 0.00e+00 | 2.40e-01 | True |
| p3 | 0.00e+00 | 2.40e-01 | True |

Retired-rule R3 (4-cell band) status for the record: **FAIL**

Overall verdict: **FAIL**
