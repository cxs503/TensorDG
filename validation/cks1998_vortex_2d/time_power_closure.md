# CKS1998 §5.3 — power-calibration closure

production CFL 0.2, beta 4, eta_max 0.05, admissible CFL <= 0.5, order tol 0.1

## p1 (primary mesh 10)

* mesh verdicts: {'10': 'GATE_VACUOUS_CERTIFIED'}
* mesh independent: True
* closure verdict: **GATE_VACUOUS_CERTIFIED**

### mesh 10

| CFL | e(CFL) | e(CFL/beta) | eta | leg |
|---:|---:|---:|---:|---|
| 0.05 | 4.140572e-03 | 4.140572e-03 | 2.659e-08 | SILENT |
| 0.1 | 4.140571e-03 | 4.140572e-03 | 3.179e-07 | SILENT |
| 0.2 | 4.140564e-03 | 4.140572e-03 | 1.881e-06 | SILENT |
| 0.4 | 4.140516e-03 | 4.140571e-03 | 1.329e-05 | SILENT |

* fit: s = 2.9458912430785014, r2 = 0.9948456807275667, n = 4, crossing = 6.351690662265041
* jackknife exponent range [2.6925340111041476, 3.0722804445525296], crossing range [5.303872587374206, 8.605313106286767]
* three-level q_obs: 3.072280444552533, agreement CONSISTENT (gap 0.04113856262638156)
* window sup: {'status': 'OK', 'admissible_cfl_max': 0.5, 'eta_sup': 1.3286494239967545e-05, 'at_cfl': 0.4, 'n_admissible': 4, 'sup_over_eta_max': 0.0002657298847993509, 'below_proposed_1e4': True}
* second refinement: {'status': 'OK', 'errors': {'refined2': 0.004140572215452991, 'refined': 0.004140572105338447, 'base': 0.004140564315274795}, 'eta_one_level': 1.8814014366836037e-06, 'omega_second_level': 2.6594040972495397e-08, 'omega_over_eta': 0.014135229437994595, 'temporal_asymptote_reached': True}
* drift leg: {'status': 'INSUFFICIENT_MESHES', 'design_order': 2}
