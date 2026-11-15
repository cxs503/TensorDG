# CASE-LAX-DG · reference-projection floor of the acceptance metric

> identity `||u_h-u||^2 = ||u_h-Pi u||^2 + ||Pi u-u||^2`; tolerance 3.00 %; digest in `metrics.sha256`

## Measured legs (committed profiles, no re-solve)

| p | N | total L2 quad | total L2 exact | solver leg | reference floor | floor / tolerance |
|---|---|---|---|---|---|---|
| 1 | 400 | 4.93486e-02 | 4.88715e-02 | 4.16161e-02 | 2.56227e-02 | 85.41 % |
| 1 | 800 | 3.77126e-02 | 3.80275e-02 | 3.50298e-02 | 1.47989e-02 | 49.33 % |
| 2 | 400 | 3.52587e-02 | 3.44463e-02 | 2.46985e-02 | 2.40112e-02 | 80.04 % |
| 2 | 800 | 2.57033e-02 | 2.63363e-02 | 2.38338e-02 | 1.12050e-02 | 37.35 % |
| 3 | 200 | 4.19998e-02 | 4.34844e-02 | 3.28126e-02 | 2.85347e-02 | 95.12 % |
| 3 | 400 | 3.11556e-02 | 3.02507e-02 | 2.41102e-02 | 1.82704e-02 | 60.90 % |
| 3 | 800 | 2.31490e-02 | 2.40498e-02 | 2.21343e-02 | 9.40561e-03 | 31.35 % |

## Reference floor ladder (solver-free)

| p | N=100 | N=200 | N=400 | N=800 | N=1600 | exponent |
|---|---|---|---|---|---|---|
| 1 | 5.8562e-02 | 4.0467e-02 | 2.5623e-02 | 1.4799e-02 | 1.1216e-02 | -0.622 |
| 2 | 4.1385e-02 | 3.2503e-02 | 2.4011e-02 | 1.1205e-02 | 9.0200e-03 | -0.593 |
| 3 | 3.4911e-02 | 2.8535e-02 | 1.8270e-02 | 9.4056e-03 | 8.6591e-03 | -0.562 |

## Hypothesis H

* statement: reference floor / solver leg >= 0.25 at p3.N400
* floor / solver: density 0.758, velocity 0.511, pressure 0.516
* supported: **True** -- the registered metric is co-dominated by the reference discretisation at the pinned mesh

## Invariants

* Pythagoras residual, density (worst): 6.761e-16
* decomposition residual incl. cross term (worst): 1.084e-18
* cross term (worst, velocity/pressure): 1.383e-04
* conserved-field reconstruction residual (worst): 8.882e-15
* reuse vs committed pool p3.N400: 0.000e+00
* quadrature saturation (20/40/80 pt): 2.377e-16
* split-position sensitivity: 1e-06 -> 8.501e-03, 1e-09 -> 1.030e-05, 1e-12 -> 1.030e-08
* wall time: 54.2 s
