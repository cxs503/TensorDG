# DMR refinement ladder — gate verdicts

| gate | verdict | pass_if | evidence |
|---|---|---|---|
| G1 | **PASS** | abs(rho_err) <= 0.03 and abs(p_err) <= 0.03 for all P1 rungs | `{"64": {"rho": -0.01823028593991682, "p": -0.023780359143949232}, "128": {"rho": -0.010924677208807408, "p": -0.014764540035265563}}` |
| G2 | **PASS** | err(384x96) < err(64x16) | `{"coarse": {"nx": 64, "abs_rel": 0.01823028593991682}, "fine": {"nx": 128, "abs_rel": 0.010924677208807408}, "rate": 0.7387465369810239}` |
| G3 | **PASS** | <= 2.0 cells at every P1 rung | `{"64": 1.4241193203508526, "128": 1.1108186921336696}` |
| G4 | **FAIL** | front_error_phys(384x96) <= 0.5 * front_error_phys(64x16) | `{"coarse": {"nx": 64, "abs_phys": 0.22735170051988374}, "fine": {"nx": 128, "abs_phys": 0.17246042322168365}, "ratio": 0.7585622752208119}` |
| G5 | **PASS** | 0.5 <= smear_width_cells(384x96) / smear_width_cells(64x16) <= 2.0 | `{"64": 5.812920045170102, "128": 6.221540853202207, "ratio_finest_over_coarsest": 1.070295274123308}` |
| G6 | **FAIL** | abs(err(384x96)) < abs(err(64x16)) and abs(err(384x96)) <= 0.03 | `{"64": {"rel_error_rho": -0.21622158716413503, "offset_cells": 1.6}, "128": {"rel_error_rho": -0.12997067944191218, "offset_cells": 3.2}}` |
| G7 | **FAIL** | observed rate >= 0.5 (least squares on log-log over all rungs) | `{"l1_rel": {"64": 0.17607309725032466, "128": 0.15923676929941077}, "rate": 0.14500098570807762}` |
| G8 | **FAIL** | offset_cells(64x16) < smear_width_cells(64x16) and offset_cells(384x96) > smear_width_cells(384x96) | `{"coarse": {"offset_cells": 1.6, "smear_width_cells": 5.812920045170102}, "fine": {"offset_cells": 3.2, "smear_width_cells": 6.221540853202207}}` |

## P1 ladder

| grid | dx | steps | interior rho err | interior p err | top-window rho err | smear width [cells] | max front err [cells, y<=0.8] | front err @y~0.90 [cells] | L1 rel (undisturbed) |
|---|---|---|---|---|---|---|---|---|---|
| 64x16 | 0.06250 | 150 | -1.8230% | -2.3780% | -32.4898% | 5.81 | 3.64 | -3.64 | 0.1761 |
| 128x32 | 0.03125 | 313 | -1.0925% | -1.4765% | -33.3289% | 6.22 | 5.52 | -5.52 | 0.1592 |
