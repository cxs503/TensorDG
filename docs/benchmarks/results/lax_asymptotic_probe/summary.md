# CASE-LAX-DG · p3 order probe (N = 400 -> 800)

> Question: p3 C3/R3 order FAIL: intrinsic scheme property or pre-asymptotic resolution artefact?
> Resolution: H1 supported as a trend: order still escalating (x1.69) but 0.463 < 3.40
> Verdict (pool gates on this ladder): **FAIL** · metrics sha256 `dd9810aa57878b68812b1fe525d2e158326b935f27ebfec4886e46fafbd56243`

## Ladder (programmatic, zero transcription)

| p | N | steps | whole-field L1 rho | analytic L1 rho | band cells | ripple wavelength (cells) |
|---|---|---|---|---|---|---|
| 1 | 400 | 2199 | 7.21833e-03 | 1.30826e-04 | 8.0 | 2.02 |
| 1 | 800 | 4390 | 4.23667e-03 | 3.10623e-05 | 16.0 | 2.78 |
| 2 | 400 | 3662 | 4.52816e-03 | 6.12389e-04 | 8.0 | 1.04 |
| 2 | 800 | 7341 | 2.54812e-03 | 1.51985e-04 | 16.0 | 1.30 |
| 3 | 200 | 2547 | 1.21474e-02 | 7.34045e-03 | 4.0 | 1.04 |
| 3 | 400 | 5109 | 8.68554e-03 | 6.07161e-03 | 8.0 | 1.04 |
| 3 | 800 | 10250 | 6.16475e-03 | 4.40521e-03 | 16.0 | 1.04 |

## Orders (analytic region, nested band)

| p | pair | order rho | requirement |
|---|---|---|---|
| 1 | 400 -> 800 | 2.074 | 1.70 |
| 2 | 400 -> 800 | 2.011 | 2.55 |
| 3 | 200 -> 400 | 0.274 | 3.40 |
| 3 | 400 -> 800 | 0.463 | 3.40 |

## p3 resolution

* r3: previous pair 200->400 order 0.274; finest pair 400->800 order 0.463 (escalation x1.691)
* c3: previous pair 200->400 order 0.484; finest pair 400->800 order 0.495 (escalation x1.022)

## Reuse check
* worst relative difference vs committed pool metrics: 0.000e+00 (docs/benchmarks/results/lax_case_pool/metrics.json)

Wall time: 510.0 s
