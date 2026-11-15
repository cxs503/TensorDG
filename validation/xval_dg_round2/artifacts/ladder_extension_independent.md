# Ladder extension (3 -> 4 levels): independent recompute

Every number below is produced by `validation/xval_dg_round2/close_ladder_extension.py`
from the committed raw profiles, **without importing the engine** (`tensordg.certified_reference`).  It exists to disagree with the engine if it can.

## L0 successive differences

| levels | rel L1 % |
|---|---:|
| 800 -> 1600 | 0.230839 |
| 1600 -> 3200 | 0.094893 |
| 3200 -> 6400 | 0.041976 |

strictly decreasing: **True**

## L1 / L2 order and band

| quantity | value |
|---|---:|
| p3 ([800,1600,3200]) | 1.282509 |
| p4 ([1600,3200,6400]) | 1.176735 |
| relative order change | 0.0825 |
| U3 (3-level band) | 0.082797 % |
| U4 (4-level band) | 0.041622 % |
| U4 share of gate | 0.008 % |

## L3 band coverage (the band's own falsifier)

| quantity | value |
|---|---:|
| shift rel L1(cert3, cert4) on N=1600 | 0.059387 % |
| U3 | 0.082797 % |
| coverage ratio shift/U3 | 0.7173 |
| covered | **True** |

## L4 claim stability

| leg | vs 3-level ref % | vs 4-level ref % | delta pp |
|---|---:|---:|---:|
| DG P1 N=200 CFL=0.4 | 2.455701 | 2.448436 | -0.007265 |
| DG P2 N=200 CFL=0.4 | 2.420304 | 2.413098 | -0.007206 |
| DG P3 N=200 CFL=0.4 | 2.426629 | 2.419353 | -0.007276 |
| DG P1 N=400 CFL=0.4 | 1.547928 | 1.545491 | -0.002437 |
| DG P2 N=400 CFL=0.4 | 1.516026 | 1.513771 | -0.002254 |
| DG P3 N=400 CFL=0.4 | 1.522152 | 1.520666 | -0.001486 |
| DG P1 N=800 CFL=0.4 | 0.578643 | 0.580254 | +0.001611 |
| DG P2 N=800 CFL=0.4 | 0.566719 | 0.564234 | -0.002485 |
| DG P3 N=800 CFL=0.4 | 0.670702 | 0.665575 | -0.005128 |
| DG P2 N=400 CFL=0.1 | 1.519713 | 1.517166 | -0.002547 |
| DG P2 N=400 CFL=0.2 | 1.518092 | 1.515633 | -0.002458 |

worst leg: DG P1 N=200 CFL=0.4 = 2.448436 %; max |delta| = 0.007276 pp

## Independent vs engine

| quantity | independent | engine | rel diff | agree |
|---|---:|---:|---:|:--:|
| L0.d1_percent | 0.23083911384330436 | 0.23083911384330436 | 0.000e+00 | yes |
| L0.d2_percent | 0.09489332553025007 | 0.09489332553025007 | 0.000e+00 | yes |
| L0.d3_percent | 0.041976204690910957 | 0.041976204690910957 | 0.000e+00 | yes |
| L1.band_U4_percent | 0.04162173159427086 | 0.04162173159427086 | 0.000e+00 | yes |
| L2.p4 | 1.1767348864087728 | 1.1767348864087728 | 0.000e+00 | yes |
| L3.shift_percent | 0.05938674280373328 | 0.05938674280373328 | 0.000e+00 | yes |

decisions identical: **True**; worst leg abs diff 8.295e-04 pp; all agree: **False**

## Decision

| gate | independent |
|---|:--:|
| L0 | PASS |
| L1 | PASS |
| L2 | PASS |
| L3 | PASS |
| L4 | PASS |

accepted (independent) = **True**
