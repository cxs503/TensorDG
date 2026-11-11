# CASE-SHU-OSHER-DG - ref vs got (Shu-Osher shock/entropy-wave, TensorDG)

Reference: independent FV-WENO5/HLLC/SSP-RK3 solution, N=3200 (`benchmarks/fv_weno5_1d.py`); acceptance line: relative L1 < 5%.

| configuration | steps | wall [s] | rel L1 [%] | rel L2 [%] | <5% | rho_min | rho_max | shock x_s | x_s vs RH [%] | unshocked sine [%] | mass err [%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DG P1 N=200 WENO CFL=0.4 | 1274 | 22.5 | 2.462 | 5.277 | PASS | 0.8009 | 4.1314 | 2.4064 | 0.714 | 0.0347 | 0.0000 |
| DG P1 N=400 WENO CFL=0.4 | 2576 | 48.5 | 1.544 | 3.658 | PASS | 0.8001 | 4.3462 | 2.4026 | 0.554 | 0.0086 | 0.0000 |
| DG P1 N=800 WENO CFL=0.4 | 5174 | 111.4 | 0.568 | 1.470 | PASS | 0.8000 | 4.5677 | 2.4005 | 0.465 | 0.0021 | 0.0000 |
| DG P2 N=200 WENO CFL=0.4 | 2122 | 35.6 | 2.426 | 5.247 | PASS | 0.8009 | 4.1376 | 2.4068 | 0.729 | 0.0348 | 0.0000 |
| DG P2 N=400 WENO CFL=0.4 | 4281 | 83.9 | 1.514 | 3.613 | PASS | 0.8001 | 4.3540 | 2.4024 | 0.547 | 0.0086 | 0.0000 |
| DG P2 N=800 WENO CFL=0.4 | 8607 | 192.8 | 0.556 | 1.434 | PASS | 0.8000 | 4.5712 | 2.4003 | 0.458 | 0.0021 | 0.0000 |
| DG P3 N=200 WENO CFL=0.4 | 2969 | 65.1 | 2.432 | 5.266 | PASS | 0.8009 | 4.1391 | 2.4071 | 0.742 | 0.0348 | 0.0012 |
| DG P3 N=400 WENO CFL=0.4 | 5991 | 126.3 | 1.520 | 3.610 | PASS | 0.8001 | 4.3556 | 2.4023 | 0.542 | 0.0086 | 0.0189 |
| DG P3 N=800 WENO CFL=0.4 | 12049 | 295.4 | 0.662 | 1.444 | PASS | 0.8000 | 4.5729 | 2.4002 | 0.454 | 0.0020 | 0.3019 |

Reference solution residual on the closed-form anchors:

| anchor | reference | exact | error |
|---|---|---|---|
| quiet-zone rho (x<=-3) | 3.857143 | 3.857143 | 4.37e-11% |
| unshocked sine (x>=2.6) | - | closed form | 0.0000% |
| shock position | 2.397401 | 2.389367 | 0.336% |
| global mass M(1.8) | 31.089152 | 31.089152 | 1.85e-06% of M0 |

Reference self-convergence (N=1600 vs N=3200): L1 0.101%, L2 0.358%.

Best configuration: DG P2 N=800 WENO CFL=0.4 - rel L1 0.556% vs the 5% gate (PASS).
