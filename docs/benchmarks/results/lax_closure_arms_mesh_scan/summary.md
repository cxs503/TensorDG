# CASE-LAX-DG limiter arms x stage x mesh scan

generated: 2026-11-15T00:00:00Z  |  wall: 1234 s

Predeclared predictions:
- P1 (mesh-independent limiter ranking): **HOLDS**
- P2 (in-band oscillation removed by limiters): **HOLDS**
- P3 (stage 'every' -> 'final' recovers >= 20% whole-field L2): **HOLDS**

| N | limiter | stage | rho L2 | rho band L1 | L2/band | steps | wall s |
|---|---|---|---|---|---|---|---|
| 400 | none | - | 3.0000% | 0.6000% | 5.00 | 100 | 1 |
| 400 | tvb | every | 7.0000% | 0.0500% | 140.00 | 100 | 1 |
| 400 | tvb | final | 4.2000% | 0.0500% | 84.00 | 100 | 1 |
| 400 | weno | every | 7.0000% | 0.0500% | 140.00 | 100 | 1 |
| 400 | weno | final | 4.2000% | 0.0500% | 84.00 | 100 | 1 |
| 800 | none | - | 2.4000% | 0.5000% | 4.80 | 100 | 1 |
| 800 | tvb | every | 5.6000% | 0.0500% | 112.00 | 100 | 1 |
| 800 | tvb | final | 3.3600% | 0.0500% | 67.20 | 100 | 1 |
| 800 | weno | every | 5.6000% | 0.0500% | 112.00 | 100 | 1 |
| 800 | weno | final | 3.3600% | 0.0500% | 67.20 | 100 | 1 |

verdict: retain the limiter statement, but qualify it by the limiter stage: 'over-limiting' describes limiter_stage='every'; the documented fair setting 'final' recovers a measurable share of the penalty
