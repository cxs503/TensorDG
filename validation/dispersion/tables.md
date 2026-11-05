| degree | phase-err order (pair) | ppw @1% | ppw @0.1% | $\|\lambda\|_{max}h/a$ | $(2p+1)$ | $cfl_{max}$ | $cfl_{max}/0.08$ |
|---|---|---|---|---|---|---|---|
| 0 | 2.00 (N=32->64) | 1979.83 | 20075.55 | 2.000 | 1 | 1.393 | 17.4 |
| 1 | 3.99 (N=32->64) | 12.94 | 28.17 | 5.990 | 3 | 1.395 | 17.4 |
| 2 | 5.99 (N=16->32) | 3.79 | 6.05 | 11.842 | 5 | 1.176 | 14.7 |
| 3 | 7.98 (N=8->16) | 2.00 (Nyquist) | 2.83 | 19.125 | 7 | 1.019 | 12.7 |

| anchor | value |
|---|---|
| $p=0$ symbol vs closed-form upwind FTBS, $\max|\lambda-\lambda_{exact}|$ | 4.441e-15 |
| $\max \mathrm{Re}\lambda$ (p0) | -6.149e-01 |
| $\max \mathrm{Re}\lambda$ (p1) | -6.578e-04 |
| $\max \mathrm{Re}\lambda$ (p2) | -2.541e-07 |
| $\max \mathrm{Re}\lambda$ (p3) | -5.001e-11 |
| branch-tracking ambiguities (p0, N=32) | 0 |
| branch-tracking ambiguities (p1, N=32) | 0 |
| branch-tracking ambiguities (p2, N=32) | 0 |
| branch-tracking ambiguities (p3, N=32) | 0 |
| block symbol vs full-operator projection (p0) | 4.441e-15 |
| block symbol vs full-operator projection (p1) | 5.859e-14 |
| block symbol vs full-operator projection (p2) | 2.050e-13 |
| block symbol vs full-operator projection (p3) | 4.823e-13 |
| RK4 step: spectral operator vs production solver (p0_n32) | 0.000e+00 |
| RK4 step: spectral operator vs production solver (p1_n32) | 1.397e-17 |
| RK4 step: spectral operator vs production solver (p2_n32) | 1.397e-17 |
| RK4 step: spectral operator vs production solver (p3_n32) | 1.234e-17 |

| degree | 2-D ndofs | $\|\lambda\|_{max}$ | $\|\lambda\|_{max}/\mathrm{wave\_rate}$ | $(2p+1)$ | $cfl_{max}$ |
|---|---|---|---|---|---|
| 0 | 36 | 18.000 | 2.000 | 1 | 1.393 |
| 1 | 144 | 54.000 | 6.000 | 3 | 1.393 |
| 2 | 324 | 106.581 | 11.842 | 5 | 1.176 |
| 3 | 576 | 172.412 | 19.157 | 7 | 1.018 |

| CFL | dt | steps | terminal $L_2$ error |
|---|---|---|---|
| 0.08 | 1.111e-03 | 180 | 2.936e-03 |
| 1.0 | 1.333e-02 | 15 | 2.936e-03 |
| 2.0 | 2.500e-02 | 8 | 1.752e+01 |
| 4.0 | 5.000e-02 | 4 | 5.159e+04 |
