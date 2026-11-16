# XVAL-DG-ROUND2-CERTIFIED-REFERENCE — DG-side time axis

Question: is the shipped DG P2 N=400 leg error at CFL=0.4 time-converged?

## DG legs: is the shipped leg error time-converged?

Legs compared: **CFL=0.4, CFL=0.2, CFL=0.1** (DG P2 N=400), judged against the certified reference (verdict CERTIFIED, U = 0.041622 %, stored on N=3200).

| CFL | err vs certified ref % | decision (3 %) |
|---|---|---|
| 0.4 | 1.513771 | PASS |
| 0.2 | 1.515633 | PASS |
| 0.1 | 1.517166 | PASS |

| temporal pair | rel L1 % |
|---|---|
| 0.4 -> 0.2 | 5.065128e-03 |
| 0.2 -> 0.1 | 2.501554e-03 |

* **D0** resolvable: `False` -- time error below the band: bounded by 0.041622 %, not measurable
* **D1** R_leg = delta_t / err = 0.001649 (state CLEAN, bound-only True)
* **D2** decisions {'0.4': 'PASS', '0.2': 'PASS', '0.1': 'PASS'}, max move 0.003395 pp
* **D3** temporal order 1.0178 (usable False: differences monotone True, errors monotone False)

**DG_AXIS_CLEAN** -- accepted = `True`

## Predeclared thresholds

* `POLLUTION_MAX = 0.1` (D1 CLEAN), `POLLUTION_MARGINAL = 0.25` (D1 MARGINAL ceiling)
* `LEG_STABILITY_MAX_PP = 0.5` (D2), `P_MIN_T = 0.5` (D3)
* `band` = Roache GCI of the certified reference (Fs = 1.25), i.e. the reference's own uncertainty; D0 asks whether the time step change is large enough to be distinguishable from it at all.

