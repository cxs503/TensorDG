# Cross-family convergence-order protocol (TensorDG)

> Anchor served: `GAP-DIFFLBM-XSOLVER-ORDER` ("JOUR-DIFFLBM 收敛判据重述：跨求解器口径替代对表口径").
> Code: `tensordg/xfamily_order.py`, `benchmarks/xfamily_order_protocol.py`, `tests/test_xfamily_order.py`.
> Artifacts: `outputs/xfamily_order_protocol/{results.json, table.md, figures/xfamily_order.png}`.

## 1. Why a protocol is needed

The anchor replaces a *compare-to-table* gate (max-point < 3% against Ghia 1982, which
carries its own discretisation error and therefore an error floor) with a
*cross-solver* gate: two unrelated discretisations must agree to < 3% on every grid of a
ladder **and** the disagreement must decay with observed order p >= 1.8.

That replacement rests on two assumptions that no previous round checked:

* **A1 distinctness** – the two families are genuinely different discretisations;
* **A2 asymptotic range** – both families are inside the range where their own error is
  dominated by the leading truncation term.

This repository cannot produce the cavity numbers (those solvers live elsewhere), so the
deliverable here is the *checker* plus a controlled experiment that falsifies the naive
reading "observed cross-family order = min(p_A, p_B)".

## 2. Method

Model problem (exactly solvable, same one the SIPG benchmark already uses):

`-Delta u = 2*pi^2 sin(pi x) sin(pi y)` on `(0,1)^2`, `u = 0` on the boundary.

Five families, all sampled on the *same* node set `x_i = i/n` (rule R1):

| family | discretisation | theoretical order |
|---|---|---|
| `dg_p1` | SIPG DG degree 1 (`tensordg.poisson_2d`, sparse backend), jump-averaged trace | 2 |
| `dg_p2` | SIPG DG degree 2 | 3 |
| `dg_p3` | SIPG DG degree 3 | 5 (nodal superconvergence) |
| `fd2` | 5-point finite differences | 2 |
| `fd4` | 9-point compact Mehrstellen FD: `L9 u = (1 - h^2/12 D)(-F)` | 4 |

Measured self-orders on n = 8/16/32/64 (LSQ fit of max-point error):
`dg_p1 $2.002`, `dg_p2 $3.056`, `dg_p3 $4.997`,
`fd2 $2.004`, `fd4 $4.007` — the families are what they claim to be,
so the experiment can falsify the order statement rather than assume it.

## 3. Protocol rules (machine-checked in `cross_family_verdict`)

| rule | check | consequence |
|---|---|---|
| R1 shared grid | one resolution sequence for all families | shape mismatch raises |
| R2 self-converged | `converged_a/b` flags | unconverged -> verdict not creditable |
| R3 distinctness | difference > `signal_levels` x round-off floor | degenerate pair -> order `not_run` |
| R4 asymptotic window | only levels above the floor enter the fit | <3 usable levels -> `not_run` |
| R5 attribution | observed p must lie in the bracket `[min(p), max(p)]` and not be an alias | alias / out-of-bracket -> **no credit** |

## 4. Results

![cross-family order](figures/xfamily_order.png)

| pair | p bracket | grids | diff % (finest->coarsest) | p_finest | p_lsq | attribution | gate<3% | gate p>=1.8 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| dg_p1 vs fd2 | 2-2 | 8-16-32-64 | 2.51e-05 / 0.000201 / 0.00161 / 0.013 | 3.00 | 3.00 | alias_pair_shared_error | True | True | not admissible/FAIL |
| dg_p2 vs fd2 | 2-3 | 8-16-32-64 | 0.0201 / 0.0803 / 0.322 / 1.29 | 2.00 | 2.00 | dominant_family_controls | True | True | PASS |
| dg_p3 vs fd2 | 2-5 | 8-16-32-64 | 0.0201 / 0.0804 / 0.322 / 1.3 | 2.00 | 2.00 | dominant_family_controls | True | True | PASS |
| dg_p2 vs fd4 | 3-4 | 8-16-32-64 | 1.05e-05 / 0.000168 / 0.00269 / 0.0435 | 4.00 | 4.01 | dominant_family_controls | True | True | PASS |
| dg_p3 vs fd4 | 4-5 | 8-16-32-64 | 9.68e-06 / 0.000155 / 0.00249 / 0.0403 | 4.00 | 4.01 | dominant_family_controls | True | True | PASS |

Findings:

1. **Dominant-family regime (creditable).** When one family's own error dominates in
   magnitude the cross-family order is that family's order and lies in the bracket.
   `dg_p2/dg_p3 vs fd2` -> 2.00 (fd2 owns the difference); `dg_p3 vs fd4` -> 4.00.
   The triangle bound `|A-B| <= e_A + e_B` is tight here: measured
   `|A-B|/(e_A+e_B)` in [0.686, 1.000].
2. **Alias regime (not creditable).** `dg_p1 vs fd2` agree to
   `~1.00%` of either family's own error and their difference decays at
   **3.00** although *both* families are 2nd order. Reason: SIPG p=1 on a Cartesian mesh
   and the 5-point FD operator share their leading truncation term, so the difference is
   the (small) BC/penalty residual, not an accuracy measure. Both numeric gates in the
   anchor's acceptance (3%, p >= 1.8) would read **green** for this pair, yet the pair
   certifies nothing. `cross_family_verdict(...).passed` is `False` and the attribution is
   `alias_pair_shared_error`.
3. **Order alone cannot separate "both right" from "both wrong alike".** The alias regime
   is invisible to a cross-solver metric without each family's own error against an
   independent reference. Consequence for the anchor: every cross-solver number must be
   reported together with per-family own errors, otherwise the gate is unfalsifiable.
4. **Half-gate rule.** `|A-B| <= e_A + e_B` implies a `<3%` agreement *requires* each
   family's own max-point error `< 1.5%` **in the same
   normalisation** (`half_gate_budget(3.0) = 1.5`). Tuning the
   pair, or averaging the two fields, cannot buy the gate.

## 5. Applied to the anchor's published ladder (cited, not recomputed)

The cavity ladder of report **#19236** (v-normalised percentages, n = 65/97/129/161:
3.4950, 1.6746, 0.9321, 0.5569 %; lid-speed-normalised:
0.8896, 0.4256, 0.2367, 0.1506 %) is passed through the same
checker:

* order bracket for a strictly 2nd-order pair `[2, 2]`; measured `p_finest =
  2.32` -> **above_range_violation**
  (0.07 beyond the +0.25 tolerance), i.e. the coarse ladder is *not* a clean 2nd-order
  pair measurement: part of the agreement is leading-term cancellation.
* gate: `n=65` at 3.495% fails the 3% gate while
  `n >= 97` passes (1.675, 0.932, 0.557 %);
  `required_resolution_for_gate` inverts the measured decay to
  `n ≈ 78`, i.e. **n=65 is outside the asymptotic
  window** -- consistent with, and quantitatively explaining, the published
  "n>=97 all <3%" observation.
* normalisation: the two quoted percentages differ by exactly
  `3.929x` = `U_lid/max|v|`; they are the *same* absolute
  difference, so a gate stated "in percent" is meaningless without its normaliser. The
  anchor's acceptance text does not name one -- this is the second defect the checker
  exposes (`scale` is a mandatory argument).
* alias detector: **not_run** — the published ladder contains no per-family own error, so
  the agreement cannot be credited as accuracy. This matches the independent audits
  (reports #19192 / #19258) that graded the anchor L2.

## 6. Re-specified acceptance proposal (for the ledger entry)

Keep the cross-solver idea, add the three checks that make it falsifiable:

1. `max-point < 3%` **and** the normaliser named explicitly (`U_lid` recommended; the
   v-normalised metric is 3.93x stricter because `max|v| ≈ 0.255 U_lid`);
2. observed order `p >= 1.8` **and** `p` inside `[min(p_A,p_B), max(p_A,p_B)]` with
   `|A-B|/max(e_A,e_B) > 0.1` at every level (no alias);
3. each family's own error against an independent reference reported per grid (the
   anchor's own reference floor problem is solved by Richardson-extrapolated ladder, not
   by a table), so the half-gate rule `< 1.5%` per family is visible.

Under this re-spec the published ladder already satisfies (1) with the `U_lid` normaliser
and (2) for `n >= 97`; what is missing is (3).

## 7. Limitations / what was not done

* The cavity problem itself is **not** solved here; `results.json.anchor_cited` carries
  `status = "cited_not_recomputed"` and every use of it is labelled.
* `dg_p3`'s nodal order is 5 (superconvergence), not the 4 of the L2 rate; that is why the
  bracket for `dg_p3 vs fd4` is 4-5 rather than 4-4.
* The alias threshold (0.1) and the bracket tolerance (0.25) are protocol conventions, not
  derived constants; they are exposed as parameters (`alias_threshold`, tolerance in
  `cross_family_verdict`) and documented in the tests.
* One deliberately clumsy step is documented because it cost time and is easy to repeat:
  the compact FD4 right-hand side must be `-F(1 - h^2 pi^2/6)`; with the sign flipped the
  scheme solves `u = -u_exact` (O(1) error) and with the correction omitted it degrades to
  second order.

## 8. Reproduction

```
PYTHONPATH=. python benchmarks/xfamily_order_protocol.py           # ~55 s, writes outputs/
PYTHONPATH=. python benchmarks/xfamily_order_protocol.py --quick   # 3-grid smoke run
PYTHONPATH=. python -m pytest tests/test_xfamily_order.py -q       # 14 tests
```
