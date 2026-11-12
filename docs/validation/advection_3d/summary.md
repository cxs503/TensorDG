# 3-D periodic advection -- verification ladder

Verdict: **PASS**  |  commit `6d0366d80674d0953af8bcc4a6ade9c4d13eecc9`
| config: `t=0.25`, `a=[1.0, 0.5, 0.25]`, `cfl=0.08`,
torch 2.7.1, numpy 2.4.6

| degree | cells | h | relative L2 | absolute L2 | EOC | mass drift | steps |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 0.2500 | 7.5925e-02 | 8.0531e-02 | -- | 0.00e+00 | 66 |
| 1 | 8 | 0.1250 | 2.0871e-02 | 2.2138e-02 | 1.863 | 0.00e+00 | 132 |
| 1 | 16 | 0.0625 | 5.3732e-03 | 5.6991e-03 | 1.958 | 0.00e+00 | 263 |
| 2 | 4 | 0.2500 | 1.0723e-02 | 1.1374e-02 | -- | 0.00e+00 | 110 |
| 2 | 8 | 0.1250 | 1.4024e-03 | 1.4874e-03 | 2.935 | 0.00e+00 | 219 |
| 2 | 12 | 0.0833 | 4.0591e-04 | 4.3053e-04 | 3.058 | 0.00e+00 | 329 |
| 3 | 4 | 0.2500 | 1.0829e-03 | 1.1486e-03 | -- | 2.22e-16 | 154 |
| 3 | 6 | 0.1667 | 2.2084e-04 | 2.3423e-04 | 3.921 | 0.00e+00 | 230 |
| 3 | 8 | 0.1250 | 6.7216e-05 | 7.1294e-05 | 4.135 | 0.00e+00 | 307 |

| gate | measured | threshold | status |
|---|---:|---|---|
| `design_order_P1` | 1.9577e+00 | 2 +/- 15% | PASS |
| `mass_conservation_P1` | 0.0000e+00 | 1e-12 | PASS |
| `design_order_P2` | 3.0577e+00 | 3 +/- 15% | PASS |
| `mass_conservation_P2` | 0.0000e+00 | 1e-12 | PASS |
| `design_order_P3` | 4.1348e+00 | 4 +/- 15% | PASS |
| `mass_conservation_P3` | 2.2204e-16 | 1e-12 | PASS |
| `documented_case_below_3pct_P1` | 2.0871e+00 | 3.0 | PASS |
| `documented_case_below_3pct_P2` | 1.4024e-01 | 3.0 | PASS |

Figure: `advection_3d_convergence.png` (left: convergence; right: mid-plane error).
