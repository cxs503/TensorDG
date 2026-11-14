# XVAL-DG-ROUND2-CERTIFIED-REFERENCE — Lax Shock Tube Demo (Synthetic)

## D1

| verdict | band (%) | U/τ | pass |
|---|---|---|---|
| WITHHELD_BAND_CONSUMES_GATE | 1.912512 | 0.3825 | False |

## D2

| configuration | rel L1 (%) | verdict |
|---|---|---|
| DG P1 N=50 CFL=0.35 | 22.4724 | WITHHELD |
| DG P1 N=100 CFL=0.35 | 9.5460 | WITHHELD |
| DG P1 N=200 CFL=0.35 | 4.3382 | WITHHELD |
| DG P2 N=50 CFL=0.35 | 16.0603 | WITHHELD |
| DG P2 N=100 CFL=0.35 | 6.8095 | WITHHELD |
| DG P2 N=200 CFL=0.35 | 3.1333 | WITHHELD |
| DG P3 N=50 CFL=0.35 | 13.2165 | WITHHELD |
| DG P3 N=100 CFL=0.35 | 5.5972 | WITHHELD |
| DG P3 N=200 CFL=0.35 | 2.5978 | WITHHELD |

**Worst**: 22.4724 %
**All pass (< 3 %)**: False

## D3

| rel L1 (%) | tol (%) | pass |
|---|---|---|
| 0.3036 | 1.0 | True |

## D4

| ladder | verdict |
|---|---|
| dg_p1 | WITHHELD_TOO_FEW_LEVELS |
| dg_p2 | WITHHELD_TOO_FEW_LEVELS |
| dg_p3 | WITHHELD_TOO_FEW_LEVELS |

**accepted = False**
