# REPRO-DG-CKS1998 — measured vs anchor

profile `quick` · device `cpu` · torch 2.7.1 · loongarch64

## A1 linear advection, smooth data — L2 order (gate: order >= p+0.5)

| degree | h | L2 error | observed order | theory | gate |
|---:|---:|---:|---:|---:|---|
| 1 | 1/16 | 6.768e-03 | 2.013 | 2 | 1.5 |
| 1 | 1/32 | 1.668e-03 | 2.013 | 2 | 1.5 |
| 1 | 1/64 | 4.154e-04 | 2.013 | 2 | 1.5 |
| 2 | 1/16 | 2.089e-04 | 3.000 | 3 | 2.5 |
| 2 | 1/32 | 2.612e-05 | 3.000 | 3 | 2.5 |
| 2 | 1/64 | 3.265e-06 | 3.000 | 3 | 2.5 |
| 3 | 1/16 | 5.028e-06 | 3.998 | 4 | 3.5 |
| 3 | 1/32 | 3.153e-07 | 3.998 | 4 | 3.5 |
| 3 | 1/64 | 1.970e-08 | 3.998 | 4 | 3.5 |

## A2 square wave, TVB M=0 + SSP-RK3 (gate: L1 order >= 0.5)

| degree | h | L1 error | L2 error | pointwise max | overshoot % |
|---:|---:|---:|---:|---:|---:|
| 1 | 1/40 | 8.773e-02 | 1.645e-01 | 1.01675 | 1.675 |
| 1 | 1/80 | 5.517e-02 | 1.308e-01 | 1.01389 | 1.389 |
| 1 | 1/160 | 3.477e-02 | 1.040e-01 | 1.00917 | 0.917 |
| 2 | 1/40 | 8.653e-02 | 1.650e-01 | 1.01312 | 1.312 |
| 2 | 1/80 | 5.460e-02 | 1.311e-01 | 1.00370 | 0.370 |
| 2 | 1/160 | 3.444e-02 | 1.041e-01 | 1.00300 | 0.300 |
| 3 | 1/40 | 8.630e-02 | 1.652e-01 | 1.00757 | 0.757 |
| 3 | 1/80 | 5.448e-02 | 1.311e-01 | 1.00387 | 0.387 |
| 3 | 1/160 | 3.437e-02 | 1.041e-01 | 1.00251 | 0.251 |

## E2 Sod via DG vs exact (gate: relative L2 < 3 %)

| degree | cells | rho | u | p | worst |
|---:|---:|---:|---:|---:|---:|
| 2 | 200 | 1.604% | 4.173% | 1.150% | 4.173% |
| 2 | 400 | 1.113% | 2.368% | 0.594% | 2.368% |

## E3 TVD property (gate: overshoot <= 1 %, declared on the strict policy)

| degree | scheme | avg max | pointwise max | pointwise min | overshoot % |
|---:|---|---:|---:|---:|---:|
| 1 | unlimited | 1.06250 | 1.06850 | -0.06850 | 6.850 |
| 1 | tvb_M0 | 1.00149 | 1.01389 | -0.01389 | 1.389 |
| 1 | strict_all | 0.99996 | 0.99996 | 0.00004 | 0.000 |
| 2 | unlimited | 1.06875 | 1.08520 | -0.08520 | 8.520 |
| 2 | tvb_M0 | 1.00080 | 1.00370 | -0.00370 | 0.370 |
| 2 | strict_all | 0.99996 | 0.99996 | 0.00004 | 0.000 |

## Verdict

- **A1_smooth_order**: PASS
- **A2_discontinuous_order**: PASS
- **E2_sod_three_percent**: PASS
- **E3_tvd_no_new_extrema**: PASS
- **all_pass**: True
