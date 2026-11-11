# REPRO-DG-CKS1998 — reference vs got

Anchor: Cockburn & Shu 1998/2001 RKDG benchmark set (SIAM Review 43(1), JCP 141)
Profile: `quick` · rows: 54 (PASS 43 / FAIL 0 / informative 11)

`ref` provenance: *analytic* = exact solution, *paper_theorem* = proved order/floor in Cockburn & Shu, *withheld* = paper table cell not claimed (no PDF). Regenerate with `python -m benchmarks.cks1998_benchmark`.

| item | quantity | ref | got | abs diff | rel diff | tolerance | verdict | ref source |
|---|---|---:|---:|---:|---:|---|---|---|
| A1_smooth_order | L2 observed order p1 | 2 | 2.01315 | 0.0132 | 0.00658 | >= p+0.5 = 1.5 | PASS | paper_theorem |
| A1_smooth_order | L2 observed order p2 | 3 | 2.99976 | -0.000236 | 7.86e-05 | >= p+0.5 = 2.5 | PASS | paper_theorem |
| A1_smooth_order | L2 observed order p3 | 4 | 3.99769 | -0.00231 | 0.000578 | >= p+0.5 = 3.5 | PASS | paper_theorem |
| A2_discont_order | L1 observed order p1 | 0.5 | 0.667656 | 0.168 | 0.335 | >= 0.5 | PASS | paper_theorem |
| A2_discont_order | L1 observed order p2 | 0.5 | 0.664468 | 0.164 | 0.329 | >= 0.5 | PASS | paper_theorem |
| A2_discont_order | L1 observed order p3 | 0.5 | 0.664059 | 0.164 | 0.328 | >= 0.5 | PASS | paper_theorem |
| E2_sod_p2_n200 | rel L2 error, density |  | 0.0160431 |  |  | < 3% | PASS | analytic |
| E2_sod_p2_n200 | rel L2 error, velocity |  | 0.0417341 |  |  | < 3% | informative | analytic |
| E2_sod_p2_n200 | rel L2 error, pressure |  | 0.0115047 |  |  | < 3% | PASS | analytic |
| E2_sod_p2_n400 | rel L2 error, density |  | 0.0111321 |  |  | < 3% | PASS | analytic |
| E2_sod_p2_n400 | rel L2 error, velocity |  | 0.023677 |  |  | < 3% | PASS | analytic |
| E2_sod_p2_n400 | rel L2 error, pressure |  | 0.00594414 |  |  | < 3% | PASS | analytic |
| E2_sod_p2_n200 | density at x=0.102 | 1 | 1 | 0 | 0 | < 3% rel | PASS | analytic |
| E2_sod_p2_n200 | velocity at x=0.102 | 0 | 5.78803e-16 | 5.79e-16 |  | < 3% rel | informative | analytic |
| E2_sod_p2_n200 | pressure at x=0.102 | 1 | 1 | 0 | 0 | < 3% rel | PASS | analytic |
| E2_sod_p2_n200 | density at x=0.302 | 0.869561 | 0.862163 | -0.0074 | 0.00851 | < 3% rel | PASS | analytic |
| E2_sod_p2_n200 | velocity at x=0.302 | 0.163097 | 0.1729 | 0.0098 | 0.0601 | < 3% rel | informative | analytic |
| E2_sod_p2_n200 | pressure at x=0.302 | 0.822287 | 0.812479 | -0.00981 | 0.0119 | < 3% rel | PASS | analytic |
| E2_sod_p2_n200 | density at x=0.453 | 0.489293 | 0.49822 | 0.00893 | 0.0182 | < 3% rel | PASS | analytic |
| E2_sod_p2_n200 | velocity at x=0.453 | 0.788097 | 0.769393 | -0.0187 | 0.0237 | < 3% rel | PASS | analytic |
| E2_sod_p2_n200 | pressure at x=0.453 | 0.367621 | 0.377039 | 0.00942 | 0.0256 | < 3% rel | PASS | analytic |
| E2_sod_p2_n200 | density at x=0.552 | 0.426319 | 0.426199 | -0.000121 | 0.000283 | < 3% rel | PASS | analytic |
| E2_sod_p2_n200 | velocity at x=0.552 | 0.927453 | 0.927479 | 2.65e-05 | 2.86e-05 | < 3% rel | PASS | analytic |
| E2_sod_p2_n200 | pressure at x=0.552 | 0.30313 | 0.303145 | 1.52e-05 | 5.02e-05 | < 3% rel | PASS | analytic |
| E2_sod_p2_n200 | density at x=0.703 | 0.265574 | 0.275702 | 0.0101 | 0.0381 | < 3% rel | informative | analytic |
| E2_sod_p2_n200 | velocity at x=0.703 | 0.927453 | 0.927725 | 0.000273 | 0.000294 | < 3% rel | PASS | analytic |
| E2_sod_p2_n200 | pressure at x=0.703 | 0.30313 | 0.303244 | 0.000114 | 0.000376 | < 3% rel | PASS | analytic |
| E2_sod_p2_n200 | density at x=0.902 | 0.125 | 0.125 | -2.1e-10 | 1.68e-09 | < 3% rel | PASS | analytic |
| E2_sod_p2_n200 | velocity at x=0.902 | 0 | -1.77704e-09 | -1.78e-09 |  | < 3% rel | informative | analytic |
| E2_sod_p2_n200 | pressure at x=0.902 | 0.1 | 0.1 | -2.35e-10 | 2.35e-09 | < 3% rel | PASS | analytic |
| E2_sod_p2_n400 | density at x=0.101 | 1 | 1 | 0 | 0 | < 3% rel | PASS | analytic |
| E2_sod_p2_n400 | velocity at x=0.101 | 0 | 0 | 0 |  | < 3% rel | PASS | analytic |
| E2_sod_p2_n400 | pressure at x=0.101 | 1 | 1 | 0 | 0 | < 3% rel | PASS | analytic |
| E2_sod_p2_n400 | density at x=0.301 | 0.873497 | 0.869379 | -0.00412 | 0.00472 | < 3% rel | PASS | analytic |
| E2_sod_p2_n400 | velocity at x=0.301 | 0.157888 | 0.163239 | 0.00535 | 0.0339 | < 3% rel | informative | analytic |
| E2_sod_p2_n400 | pressure at x=0.301 | 0.827498 | 0.822045 | -0.00545 | 0.00659 | < 3% rel | PASS | analytic |
| E2_sod_p2_n400 | density at x=0.451 | 0.491778 | 0.496295 | 0.00452 | 0.00919 | < 3% rel | PASS | analytic |
| E2_sod_p2_n400 | velocity at x=0.451 | 0.782888 | 0.77351 | -0.00938 | 0.012 | < 3% rel | PASS | analytic |
| E2_sod_p2_n400 | pressure at x=0.451 | 0.370235 | 0.375017 | 0.00478 | 0.0129 | < 3% rel | PASS | analytic |
| E2_sod_p2_n400 | density at x=0.551 | 0.426319 | 0.42626 | -5.94e-05 | 0.000139 | < 3% rel | PASS | analytic |
| E2_sod_p2_n400 | velocity at x=0.551 | 0.927453 | 0.927491 | 3.82e-05 | 4.12e-05 | < 3% rel | PASS | analytic |
| E2_sod_p2_n400 | pressure at x=0.551 | 0.30313 | 0.303117 | -1.35e-05 | 4.44e-05 | < 3% rel | PASS | analytic |
| E2_sod_p2_n400 | density at x=0.701 | 0.265574 | 0.269618 | 0.00404 | 0.0152 | < 3% rel | PASS | analytic |
| E2_sod_p2_n400 | velocity at x=0.701 | 0.927453 | 0.927494 | 4.13e-05 | 4.45e-05 | < 3% rel | PASS | analytic |
| E2_sod_p2_n400 | pressure at x=0.701 | 0.30313 | 0.303136 | 6.17e-06 | 2.03e-05 | < 3% rel | PASS | analytic |
| E2_sod_p2_n400 | density at x=0.901 | 0.125 | 0.125 | -2.36e-13 | 1.89e-12 | < 3% rel | PASS | analytic |
| E2_sod_p2_n400 | velocity at x=0.901 | 0 | 1.88446e-12 | 1.88e-12 |  | < 3% rel | informative | analytic |
| E2_sod_p2_n400 | pressure at x=0.901 | 0.1 | 0.1 | -2.63e-13 | 2.63e-12 | < 3% rel | PASS | analytic |
| E3_tvd | pointwise max, p1_unlimited | 1 | 1.0685 | 0.0685 | 0.0685 | <= 1% overshoot | informative | analytic |
| E3_tvd | pointwise max, p1_tvb_M0 | 1 | 1.01389 | 0.0139 | 0.0139 | <= 1% overshoot | informative | analytic |
| E3_tvd | pointwise max, p1_strict_all | 1 | 0.999956 | -4.35e-05 | -4.35e-05 | <= 1% overshoot | PASS | analytic |
| E3_tvd | pointwise max, p2_unlimited | 1 | 1.0852 | 0.0852 | 0.0852 | <= 1% overshoot | informative | analytic |
| E3_tvd | pointwise max, p2_tvb_M0 | 1 | 1.0037 | 0.0037 | 0.0037 | <= 1% overshoot | informative | analytic |
| E3_tvd | pointwise max, p2_strict_all | 1 | 0.999956 | -4.35e-05 | -4.35e-05 | <= 1% overshoot | PASS | analytic |

## Gate verdict (from the run artifacts)

```json
{
  "A1_smooth_order": true,
  "A2_discontinuous_order": true,
  "E2_sod_three_percent": true,
  "E3_tvd_no_new_extrema": true,
  "all_pass": true
}
```
