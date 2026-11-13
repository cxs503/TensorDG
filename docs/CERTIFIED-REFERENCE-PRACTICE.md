# Certified reference for cases without an exact solution (round-2 practice, engine form)

> TensorDG · mission 17539 · branch `agent/m17539`.
> Numbers in this document are rendered by `tools/render_certified_reference.py`
> from `artifacts/m17539/certified_reference_audit.json` (written by
> `python -m tensordg.certified_reference`); none of them is transcribed.
> Anchor: feature ledger `XVAL-DG-ROUND2-CERTIFIED-REFERENCE`
> (acceptance < 15 %; the case gate used here is 5 %, the owner standard 3 %).

## 1. The problem this closes

For a case with no closed-form solution every "error < tau" statement is an
error *against some reference*, so the credibility of the reference is the whole
argument. Round 2 of CASE-SHU-OSHER (`docs/CASE-SHU-OSHER-round2.md`, m15803)
built that reference with a grid-convergence study, but as a **one-case script**:
`benchmarks/shu_osher_certified_ref.py` hard-codes the ladder, has no notion of
"is this ladder even converging?", and leaves three decisions to prose:

1. what to do when the observed order is not in the asymptotic range;
2. whether the reference's own uncertainty is small enough for the gate it is
   used at;
3. whether a candidate's PASS margin is larger than that uncertainty.

This work turns the practice into `tensordg/certified_reference.py` (case
independent, 3 pre-registered rules, machine verdicts) and **proves equivalence**
with the shipped round-2 numbers, so the two rounds stay comparable.

## 2. Method (unchanged numbers, added decisions)

* nesting ladder coarse -> fine; all differences (`rel L1`, normalised by the
  finer profile) measured on the coarsest grid, so no reconstruction ambiguity
  enters;
* observed order `p = ln(d_coarse/d_fine)/ln(r)` from the two finest differences,
  with `r` taken from the actual grid sizes;
* Richardson extrapolation `u* = u_f + (u_f - u_c)/(r^p - 1)` becomes the
  certified profile, stored on the second-finest grid (the grid the finest level
  coarsens onto exactly);
* Roache GCI with `Fs = 1.25`: `U = Fs * d_fine / (r^p - 1)`, the band
  of the **finest** level -- the certified reference carries that band.

### Pre-registered decision rules (written before the audit was run)

| # | rule | consequence |
|---|---|---|
| R1 | ladder classified `monotonic` only if `p_min <= p <= p_max` (`0.5` / `6.0`); round-off floor `d_fine <= 1e-12` refused | otherwise `WITHHELD_*`, no reference produced |
| R2 | gate validity: `U / tau <= 0.3333` | otherwise `WITHHELD_BAND_CONSUMES_GATE` -- the comparison would be vacuous |
| R3 | margin robustness: `PASS` only if `err + U < tau` | else `PASS_MARGINAL` (passes, but not resolvably) |

R1 fires on data whose differences stop shrinking; R2 is the machine form of
"the reference must be much better than the gate"; R3 stops a gate being passed
by reference noise.

### The displacement trap (why `U` and "extrapolation shift" are not comparable)

The certified profile replaces the **second-finest** level, so its distance to
that level is that level's own error, i.e. `r^p/(r^p-1)` times the medium-fine
difference -- a factor of about 1.7, larger than `U`. For the shipped case the
measured displacement is 0.171458 % against
`U = 0.082797 %` (ratio 1.000000 to the
identity). Reading that ratio as "the band is understated" is the wrong reading;
the band belongs to the finest level and the test suite pins the identity
(`tests/test_certified_reference.py::test_displacement_identity_and_the_wrong_reading`).

## 3. Which ladder may certify a reference?

Gate used for R2: **5 %**.

| ladder | levels | p (observed) | band `U` [%] | `U/tau` | classification | verdict |
|---|---|---|---|---|---|---|
| `dg_p1` | [200, 400, 800] | 0.1608 | 11.345741 | 2.269 | `order_too_low` | `WITHHELD_ORDER_TOO_LOW` |
| `dg_p2` | [200, 400, 800] | 0.1636 | 11.105507 | 2.221 | `order_too_low` | `WITHHELD_ORDER_TOO_LOW` |
| `dg_p3` | [200, 400, 800] | 0.0404 | 51.660084 | 10.332 | `order_too_low` | `WITHHELD_ORDER_TOO_LOW` |
| `fv_weno5` | [800, 1600, 3200] | 1.2825 | 0.082797 | 0.017 | `monotonic` | `CERTIFIED` |

Usable certifiers: `['fv_weno5']` (1/4).

**This is the quantitative core of the result.** The independent FV-WENO5/HLLC/
SSP-RK3 ladder is asymptotic (`p = 1.2825`, far below its
formal 5th order because the solution contains shocks) with a band of
`0.082797 %` = 1.66 % of the 5 %
gate, so it certifies. The DG families themselves do not: a fixed-degree DG ladder
refined in `N` stagnates (`p` = 0.1608, 0.1636, 0.0404),
so their bands are
11.346 %, 11.106 %, 51.660 %
-- 2.27x, 2.22x, 10.33x
the gate. The solver cannot certify itself for this case, and now there is a number for
why. The two rules fire independently: R1 refuses every DG ladder on the observed
order (`p < p_min = 0.5`), and R2 refuses them on the band alone
(`U/tau` = 2.27, 2.22, 10.33).
Either rule alone would be enough; both are recorded in the JSON `reasons`.

## 4. Shipped DG legs re-judged against the certified reference

| configuration | rel L1 [%] | rel L2 [%] | err + U [%] | decision (5 %) | decision (3 %) | robust margin vs 3 % gate |
|---|---|---|---|---|---|---|
| DG P1 N=200 CFL=0.4 | 2.455701 | 5.262588 | 2.538498 | `PASS` | `PASS` | 15.4 % |
| DG P1 N=400 CFL=0.4 | 1.547928 | 3.656721 | 1.630725 | `PASS` | `PASS` | 45.6 % |
| DG P1 N=800 CFL=0.4 | 0.578643 | 1.517108 | 0.661440 | `PASS` | `PASS` | 78.0 % |
| DG P2 N=200 CFL=0.4 | 2.420304 | 5.232715 | 2.503101 | `PASS` | `PASS` | 16.6 % |
| DG P2 N=400 CFL=0.4 | 1.516026 | 3.609797 | 1.598823 | `PASS` | `PASS` | 46.7 % |
| DG P2 N=800 CFL=0.4 | 0.566719 | 1.481060 | 0.649516 | `PASS` | `PASS` | 78.3 % |
| DG P3 N=200 CFL=0.4 | 2.426617 | 5.252309 | 2.509415 | `PASS` | `PASS` | 16.4 % |
| DG P3 N=400 CFL=0.4 | 1.522033 | 3.606569 | 1.604830 | `PASS` | `PASS` | 46.5 % |
| DG P3 N=800 CFL=0.4 | 0.669867 | 1.490799 | 0.752664 | `PASS` | `PASS` | 74.9 % |
| DG P2 N=400 CFL=0.1 | 1.519713 | 3.614589 | 1.602510 | `PASS` | `PASS` | 46.6 % |
| DG P2 N=400 CFL=0.2 | 1.518092 | 3.611538 | 1.600889 | `PASS` | `PASS` | 46.6 % |

Worst leg: DG P1 N=200 CFL=0.4 at 2.455701 %; best:
DG P2 N=800 CFL=0.4 at 0.566719 %.
Worst `err + U` over all legs = **2.538498 %**.
Decision counts at the 5 % gate: `{'PASS': 11}`; at the owner
3 % standard: `{'PASS': 11}`.

### Error verdict (explicit, item by item)

| quantity | value | reference | error | < 3 % ? |
|---|---|---|---|---|
| certified band `U` vs GCI recomputation of the legacy driver | bit-identical | round-2 published 0.082797 % | 0 (exact equality) | yes |
| observed order `p` vs legacy driver | 1.282509176116 | 1.282509176116 | 0 (exact equality) | yes |
| certified profile vs legacy `rho_certified` | | | max abs diff 0.0 | yes |
| all 11 shipped legs vs certified reference | worst 2.455701 % | certified reference | 2.455701 % | **yes** (worst 2.4557 %) |
| worst leg including the band (`err + U`) | 2.538498 % | certified reference | 2.538498 % | **yes** |

**Is it < 3 %?** Yes, for every shipped leg, including when the reference band is
added to the error: the worst leg is 2.4557 % and the worst
`err + U` is 2.5385 %, i.e. 15.4 % of
the 3 % budget still unused. No leg moved across a gate relative to the round-2
report. There is **no literature tabulated solution** for this case (Shu & Osher
1988 show the solution graphically), so no literature comparison is claimed --
this is code verification against a certified reference, not experimental
validation.

## 5. Is the band meaningful? (coverage check with a known exact solution)

On `200` synthetic ladders (`seed = 0`, levels
`[200, 400, 800]`) whose exact solution is known and whose error is a two-term
model `A N^-p + B N^-q`, the certified profile's true error was covered by its own
band in **100.0 %** of the cases; the worst
true/band ratio was 0.007842 and the median
0.000118. So on smooth ladders the band is *strongly
conservative* (the observed order is high and the extrapolation removes almost
all of the error). This does **not** transfer to the shock case: there the
observed order is ~1 and the true error is unmeasurable, which is exactly why the
band -- not a claimed exactness -- carries the argument.

## 6. Figures

### fig1 -- certified reference, its band, and a DG leg
![fig1](artifacts/m17539/fig1_certified_reference_vs_dg.png)

Top: certified reference with its `+/- U` ribbon, the raw FV level it replaces,
and DG P2 N=800. Bottom: differences to the reference, band lines dotted.
The reference band contributes only 3.4--14.6 % of any leg's error, so the leg errors are the solver's,
not the reference's; the window-resolved errors of the round-2 report locate the
solver error at the shock/contact (limiter smearing).

### fig2 -- gate validity per ladder and leg errors
![fig2](artifacts/m17539/fig2_band_share_and_legs.png)

Left: `U/tau` per ladder on a log axis against the `1/3` limit (green = usable).
Right: every leg's error with its band as an error bar, against the 3 % and 5 %
lines.

## 7. Notes, limitations, and what this does **not** claim

* The certified reference is only as good as the FV ladder's asymptotic claim:
  three levels (`800/1600/3200`) give one order estimate, no redundancy. A fourth
  level was never produced (cluster job `cj1183` remained queued), so the band is
  a three-grid Roache estimate with `Fs = 1.25`, not a four-grid verification.
* `Fs = 1.25` is Roache's factor for a three-grid study; it is not derived from
  the data, so `U` is a convention-bound uncertainty, not a confidence interval.
* `p` is measured, not assumed: the formal 5th order of WENO5 is **not** observed
  (1.2825) because the solution contains shocks. Any
  statement of the form "5th-order scheme, so the reference is 5th-order
  accurate" would be false here.
* The DG-family refusal is a statement about **this case at these grids**
  (`200/400/800`, TVB limiter `M = 1.0`), not about DG in general: on smooth
  solutions the same engine certifies at the design order (see the synthetic
  power-law tests, `p` recovered to 1e-9 ... 2e-3 relative).
* **Package-root export is deferred, honestly.** `tests/test_public_api_contract.py`
  is already RED on this branch (4 symbols of `tensordg.convergence` are not
  re-exported); this module's 11 public names join that list because five of them
  (`observed_order`, `richardson_extrapolate`, `coarsen`, `rel_l1`, `gci`) are
  owned by `tensordg.convergence` and the root must resolve the collision through
  its alias table. Nothing here is silently renamed; the intended import path is
  `from tensordg.certified_reference import certify, judge`.
* The window-resolved errors of the round-2 report (plateau / entropy zone /
  pre-shock sine) are not recomputed here; they live in
  `artifacts/m15803/benchmark_ref_vs_got.md` and are unchanged.

## 8. Reproduction

```bash
python -m tensordg.certified_reference \
    --rawdir validation/repro_shu_osher/artifacts/raw --outdir artifacts/m17539
python -m tools.render_certified_reference          # -> this document + the two figures
python -m pytest tests/test_certified_reference.py -q
```

Artifacts: `artifacts/m17539/certified_reference_audit.json` (all numbers, with
`input_sha256` of the ladder it was computed from),
`certified_reference_audit.md`, `certified_reference_fv.npz` (the certified
profile), `fig1_certified_reference_vs_dg.png`, `fig2_band_share_and_legs.png`.
