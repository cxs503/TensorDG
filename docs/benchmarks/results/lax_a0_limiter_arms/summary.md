# A0 ledger arm across limiter branches (pinned leg p3.N400)

generated_at_utc = 2026-11-14T22:56:41Z, source profiles sha256 = `17e4dae7b725c15f...`

| arm | rel L2 rho | rel L2 u | rel L2 p | worst | margin | C1/C2 | A0 |
|---|---|---|---|---|---|---|---|
| none | 3.115562e-02 | 2.248053e-02 | 1.305012e-02 | 3.115562e-02 | x1.039 | PASS | FAIL |
| tvb | 7.039187e-02 | 2.980530e-02 | 1.422937e-02 | 7.039187e-02 | x2.346 | PASS | FAIL |
| weno | 6.443167e-02 | 2.939725e-02 | 1.414371e-02 | 6.443167e-02 | x2.148 | PASS | FAIL |

tolerance = 0.03 (3.000 %)

A0 verdict: **FAIL** (credited arms: none)
