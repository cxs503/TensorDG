# CASE-DG-ADVECT-ORDER round 3 — stage/order cap and CFL budget

## A. Butcher barrier (published minimal-stage counts), stages -> order

| s | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| order (TensorDG) | 1 | 2 | 3 | 4 | 4 | 5 | 6 | 6 | 7 | 7 | 8 | 8 | 9 | 9 | 10 | 10 | 11 |
| order (published) | 1 | 2 | 3 | 4 | 4 | 5 | 6 | 6 | 7 | 7 | 8 | 8 | 9 | 9 | 10 | 10 | 11 |

table matches published: **True**; all registered tableaus within barrier: **True**; stages above 17: refused (yes).

## B. self-convergence of the extended ladder

| p | tableau | stages | tableau order | stages needed (p+1) | expected spatial order | fitted order | pairwise | 判定 |
|--:|---|---:|---:|---:|---:|---:|---|:--:|
| P1 | rk2 | 2 | 2 | 2 | 2 | 2.126 | 2.191 / 2.062 | — |
| P2 | rk3 | 3 | 3 | 3 | 3 | 3.002 | 3.004 / 3.000 | — |
| P3 | rk4 | 4 | 4 | 4 | 4 | 3.992 | 3.985 / 4.000 | — |
| P4 | ssp_rk54 | 5 | 4 | 5 | 5 | 4.989 | 4.986 / 4.992 | — |
| P5 | rk5 | 6 | 5 | 6 | 6 | 5.994 | 5.990 / 5.998 | PASS |
| P6 | rk5 | 6 | 5 | 7 | 7 | 6.993 | 6.987 / 6.999 | PASS |

## C. CFL budget (fixed mesh, exact time/space split)

| p | cfl with E_time/E_total <= 10% (max) | blow-up cfl (min) | share at cfl=0.2 |
|--:|--:|--:|--:|
| P1 | 0.2 | 1.5 | 6.732% |
| P2 | 0.6 | 1.5 | 0.343% |
| P3 | 1.0 | 1.5 | 0.012% |
| P4 | 0.6 | 1.5 | 0.102% |
| P5 | 0.8 | 1.5 | 0.001% |
| P6 | 0.6 | 1.0 | 0.064% |

## 判定

- S1_barrier: **PASS**
- S2_order_p5: **PASS**
- S3_order_p6: **PASS**
- S4_time_share_p5: **PASS**
- S4_time_share_p6: **PASS**
- S5_cfl_budget: **FAIL**

**总判定：FAIL**
