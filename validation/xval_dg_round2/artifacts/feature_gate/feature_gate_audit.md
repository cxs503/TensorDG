# Set-level feature gate — XVAL-DG-ROUND2-CERTIFIED-REFERENCE

Case: CASE-SHU-OSHER (1-D Euler, M=3 shock + entropy wave train, t = 1.8)  ·  gate 3 %  ·  band ceiling 33.3 % of gate

## 1. Cell-average extremum bias on the committed profiles

| N | h | bias `h²/24 f''` | f'' | status |
|---|---|---|---|---|
| 800 | 1.250e-02 | -1.413453e-03 | -2.171064e+02 | ok |
| 1600 | 6.250e-03 | -3.671608e-04 | -2.255836e+02 | ok |
| 3200 | 3.125e-03 | -9.227065e-05 | -2.267643e+02 | ok |
| 6400 | 1.562e-03 | -2.448332e-05 | -2.406808e+02 | ok |

Measured decay exponent(s): 1.945, 1.992, 1.914 (expected 2) → **law verified = True**

## 2. Functional certification under both estimators (Q1)

| functional | kind | raw verdict | raw p | raw band % | deconvolved verdict | deconv. p | deconv. band % | verdict changed |
|---|---|---|---|---|---|---|---|---|
| amplitude | extremum | CERTIFIED | 1.601 | 0.1169 | CERTIFIED | 1.455 | 0.1616 | **False** |
| crest | extremum | CERTIFIED | 4.703 | 2.616e-05 | WITHHELD_STAGNANT | -0.961 | — | **True** |
| l1 | integral | WITHHELD_STAGNANT | -0.7147 | — | WITHHELD_STAGNANT | -0.7147 | — | **False** |
| peak_x | phase | WITHHELD_STAGNANT | -0.585 | — | CERTIFIED | 0.9652 | 0.06336 | **True** |
| rms | integral | CERTIFIED | 1.512 | 0.07617 | CERTIFIED | 1.512 | 0.07617 | **False** |
| trough | extremum | CERTIFIED | 1.684 | 0.05934 | CERTIFIED | 1.568 | 0.07336 | **False** |

Gate inputs (usable certified references) — raw: amplitude, crest, rms, trough; deconvolved: amplitude, peak_x, rms, trough

Gate teeth check: {'accepts_identity': 'GATE_PASS', 'rejects_perturbed_verdict': 'GATE_FAIL', 'perturbation_percent': 10.0, 'has_teeth': True, 'n_reference_functionals': 4, 'identity_verdict': 'GATE_PASS'}

## 3. Feature gate vs profile gate on the 11 DG legs (Q2)

Required quotas: {'integral': 1, 'extremum': 1}

| leg | profile rel L1 % | profile gate | feature gate (raw) | feature gate (deconvolved) |
|---|---|---|---|---|
| DG P1 N=200 CFL=0.4 | 2.456 | PASS | GATE_FAIL | GATE_FAIL |
| DG P1 N=400 CFL=0.4 | 1.548 | PASS | GATE_FAIL | GATE_FAIL |
| DG P1 N=800 CFL=0.4 | 0.5786 | PASS | GATE_FAIL | GATE_FAIL |
| DG P2 N=200 CFL=0.4 | 2.42 | PASS | GATE_FAIL | GATE_FAIL |
| DG P2 N=400 CFL=0.1 | 1.52 | PASS | GATE_FAIL | GATE_FAIL |
| DG P2 N=400 CFL=0.2 | 1.518 | PASS | GATE_FAIL | GATE_FAIL |
| DG P2 N=400 CFL=0.4 | 1.516 | PASS | GATE_FAIL | GATE_FAIL |
| DG P2 N=800 CFL=0.4 | 0.5667 | PASS | GATE_FAIL | GATE_FAIL |
| DG P3 N=200 CFL=0.4 | 2.427 | PASS | GATE_FAIL | GATE_FAIL |
| DG P3 N=400 CFL=0.4 | 1.522 | PASS | GATE_FAIL | GATE_FAIL |
| DG P3 N=800 CFL=0.4 | 0.6699 | PASS | GATE_FAIL | GATE_FAIL |

**Summary**: 11/11 legs pass the profile gate; 0/11 pass the feature gate (deconvolved reference). Legs that pass the profile gate and fail the feature gate: **11** → profile gate sufficient = **False**.
