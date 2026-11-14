# A0 ledger arm: quadrature audit at p3.N400

generated_at_utc = 2026-11-14T23:00:43Z
pool profiles sha256 = `7bdcc57cb9a0c212...`

| quadrature | rel L2 rho | rel L2 u | rel L2 p |
|---|---|---|---|
| uniform, order 4 | 3.302458650e-02 | 2.602997040e-02 | 1.470504574e-02 |
| uniform, order 5 | 3.348006889e-02 | 2.283930206e-02 | 1.305916907e-02 |
| uniform, order 6 | 2.969593488e-02 | 2.105532555e-02 | 1.235322089e-02 |
| uniform, order 8 | 2.905815882e-02 | 1.936333159e-02 | 1.190527229e-02 |
| uniform, order 10 | 3.115561881e-02 | 2.248052874e-02 | 1.305011534e-02 |
| uniform, order 12 | 3.005037696e-02 | 2.077006592e-02 | 1.235935331e-02 |
| uniform, order 16 | 3.069950570e-02 | 2.163284545e-02 | 1.271009681e-02 |
| uniform, order 20 | 2.969007378e-02 | 1.994602595e-02 | 1.211624515e-02 |
| aligned, order 6 | 3.025073612e-02 | 2.073564554e-02 | 1.238435227e-02 |
| aligned, order 8 | 3.025073612e-02 | 2.073631635e-02 | 1.238443720e-02 |
| aligned, order 10 | 3.025073612e-02 | 2.073631862e-02 | 1.238443730e-02 |
| aligned, order 12 | 3.025073612e-02 | 2.073631862e-02 | 1.238443730e-02 |
| aligned, order 16 | 3.025073612e-02 | 2.073631862e-02 | 1.238443730e-02 |
| aligned, order 20 | 3.025073612e-02 | 2.073631862e-02 | 1.238443730e-02 |

tolerance = 0.03
convergence gate Q1: PASS
ledger verdict on the converged quadrature: **FAIL** (worst density = 3.025074e-02, margin x1.008)
