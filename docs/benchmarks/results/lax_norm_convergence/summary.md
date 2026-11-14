# renders → docs/CASE-LAX-DG-norm-quadrature-budget.md

| field | committed q=10 L2 | corrected L2 | bias |
|---|---|---|---|
| rho | 3.115562e-02 | 3.025074e-02 | +2.99 % |
| u | 2.248053e-02 | 2.073632e-02 | +8.41 % |
| p | 1.305012e-02 | 1.238444e-02 | +5.38 % |

| arm | worst | tolerance | verdict |
|---|---|---|---|
| production_l2 (committed) | 3.115562e-02 | 3.00e-02 | FAIL |
| production_l2 (corrected) | 3.025074e-02 | 3.00e-02 | FAIL |

verdict flips: **False**
