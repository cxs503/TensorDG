# XVAL-DG-ROUND2-CERTIFIED-REFERENCE — time-discretisation pollution audit

**Verdict: `WITHHELD_MISSING_INPUTS`** (accepted = False) — question: *is the published certified reference polluted by the time step it was generated at?*

**No pollution verdict is claimed.** 6 of the CFL-refined FV-WENO5 profiles this audit consumes are not in the repository, so the real-data branch has no numbers and no pollution verdict is claimed; a verdict stored earlier in this artifact must not be read as current.

CFL series [0.3, 0.15, 0.075] (ratio 2, fixed grids), reference grid N = 3200, band = Roache GCI with Fs = 1.25.

## Missing inputs

| grid | CFL | file | producer command |
|---|---|---|---|
| 800 | 0.075 | `fv_cfl0p075_n800.npz` | `python validation/xval_dg_round2/gen_cfl_ladder.py --ncells 800 --cfl 0.075 --rawdir validation/repro_shu_osher/artifacts/raw` |
| 1600 | 0.15 | `fv_cfl0p15_n1600.npz` | `python validation/xval_dg_round2/gen_cfl_ladder.py --ncells 1600 --cfl 0.15 --rawdir validation/repro_shu_osher/artifacts/raw` |
| 1600 | 0.075 | `fv_cfl0p075_n1600.npz` | `python validation/xval_dg_round2/gen_cfl_ladder.py --ncells 1600 --cfl 0.075 --rawdir validation/repro_shu_osher/artifacts/raw` |
| 3200 | 0.15 | `fv_cfl0p15_n3200.npz` | `python validation/xval_dg_round2/gen_cfl_ladder.py --ncells 3200 --cfl 0.15 --rawdir validation/repro_shu_osher/artifacts/raw` |
| 3200 | 0.075 | `fv_cfl0p075_n3200.npz` | `python validation/xval_dg_round2/gen_cfl_ladder.py --ncells 3200 --cfl 0.075 --rawdir validation/repro_shu_osher/artifacts/raw` |
| 6400 | 0.15 | `fv_cfl0p15_n6400.npz` | `python validation/xval_dg_round2/gen_cfl_ladder.py --ncells 6400 --cfl 0.15 --rawdir validation/repro_shu_osher/artifacts/raw` |

The eleven shipped DG legs and the CFL = 0.3 ladder are committed, so the audit can say *what* it would compute -- it cannot say *what the answer is* until the profiles above exist. Total cost of the table: ~1.2 CPU-hours (see `validation/xval_dg_round2/gen_cfl_ladder.py`, shardable).

The DG-side axis below needs none of the missing FV levels (the CFL = 0.3 ladder and the three shipped P2/N=400 legs are committed), so it is reported even while the FV-side claim is withheld.

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

