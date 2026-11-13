# CASE-LAX-DG · 机制对照（程序化注入）

generated_at_utc = 2026-11-13T00:47:10Z

| mechanism | C3 | R3 | verdict |
|---|---|---|---|
| none | FAIL | FAIL | FAIL |
| av-every | FAIL | FAIL | FAIL |
| tvb-every | FAIL | FAIL | FAIL |
| tvb-final | FAIL | FAIL | FAIL |

| mechanism | degree | density order | analytic order | ripple (coarse->fine) |
|---|---|---|---|---|
| none | p2 | 1.027 | 1.667 | n/a -> n/a |
| none | p3 | 0.484 | 0.274 | n/a -> n/a |
| av-every | p2 | 0.868 | 0.924 | 6.162e-03 -> 4.178e-03 |
| av-every | p3 | 0.859 | 0.332 | 8.567e-03 -> 7.315e-03 |
| tvb-every | p2 | 0.763 | 2.011 | 5.896e-02 -> 1.266e-02 |
| tvb-every | p3 | 0.759 | 1.972 | 5.877e-02 -> 1.296e-02 |
| tvb-final | p2 | 0.758 | 1.976 | 6.455e-02 -> 1.573e-02 |
| tvb-final | p3 | 0.753 | 1.852 | 5.450e-02 -> 1.051e-02 |
