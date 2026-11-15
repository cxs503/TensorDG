# CASE-LAX-DG · temporal budget at the certified legs

> Generated from `metrics.json` (measurement run 2026-11-15T05:27:02Z,
> wall 1664.2 s). Rule: T1 temporal share of the baseline-CFL whole-field relative L2 error < 5 % for every leg and every field (order-identifiable -> temporal_share, else the measured 4x-dt bound); T2 p3.N800 max-field rel L2 < ledger tolerance at every CFL level; T3 p3.N400 max-field rel L2 >= ledger tolerance at every CFL level (the FAIL is not a dt artifact); T4 baseline-CFL L2 reproduces the committed resolution artifact; T5 steps and dt_mean halve with the CFL divisor (ratio in [1.9, 2.1]). Overall PASS iff T1-T5 hold.

| leg | rel L2 ρ @ cfl=0.35 | rel L2 ρ @ cfl=0.08750 | temporal share (worst field) | T1 |
|---|---|---|---|---|
| p3.N400 | 3.11556e-02 | 3.13361e-02 | 2.006e-02 | PASS |
| p3.N800 | 2.31490e-02 | 2.32990e-02 | 1.851e-02 | PASS |

* T1 temporal budget: **PASS**
* T2 resolution PASS is dt-robust: **PASS**
* T3 pinned FAIL is not a dt artifact: **PASS**
* T4 reuse of committed baseline: **PASS**
* T5 dt refinement actually refines: **PASS**

Overall: **PASS**
