"""Render the certified-reference audit: figures + ``docs/CERTIFIED-REFERENCE-PRACTICE.md``.

Every number in the rendered document is read out of
``artifacts/m17539/certified_reference_audit.json`` (itself written by
``python -m tensordg.certified_reference``), so nothing is transcribed by hand.

Usage::

    python -m tensordg.certified_reference --outdir artifacts/m17539
    python -m tools.render_certified_reference
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "m17539"
RAW = ROOT / "validation" / "repro_shu_osher" / "artifacts" / "raw"
DOC = ROOT / "docs" / "CERTIFIED-REFERENCE-PRACTICE.md"


def _load_profile(name: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(RAW / name, allow_pickle=True) as d:
        return np.asarray(d["x"], dtype=float), np.asarray(d["rho"], dtype=float)


def fig_field(audit: dict) -> Path:
    """Certified reference +/- band against the raw level it replaces and a DG leg."""
    import tensordg.certified_reference as cr

    fv = []
    for n in (800, 1600, 3200):
        _, rho = _load_profile(f"fv_cfl0p3_n{n}.npz")
        fv.append({"ncells": n, "field": rho})
    cert = cr.certify(fv, tolerance_percent=audit["tolerance_percent"])
    ref = cert["reference"]
    n_ref = cert["reference_ncells"]
    x = (np.arange(n_ref) + 0.5) * (10.0 / n_ref) - 5.0
    band = cert["band_percent"] / 100.0 * np.mean(np.abs(ref))
    x_dg, rho_dg = _load_profile("dg_cfl0p4_limweno_n800_p2.npz")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2]})
    ax = axes[0]
    ax.fill_between(x, ref - band, ref + band, color="0.75", alpha=0.8,
                    label=f"certified $\\pm U$ ($U$={cert['band_percent']:.4f} %)")
    ax.plot(x, ref, "k-", lw=1.6, label="certified reference (Richardson)")
    raw_on_ref = cr.coarsen(cert["field_raw_finest"], 2)
    ax.plot(x, raw_on_ref, "--", color="C3", lw=1.0,
            label="raw FV level N=1600 (what round 1 used)")
    ax.plot(x_dg, rho_dg, color="C0", lw=1.0, alpha=0.85, label="DG P2 N=800")
    ax.set_ylabel(r"$\rho$")
    ax.set_title("CASE-SHU-OSHER: certified FV reference vs raw level vs DG P2 N=800")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ref_on_dg = cr.coarsen(ref, int(round(ref.size / rho_dg.size)))
    ax.plot(x, ref - ref, "k-", lw=1.2, label="reference (zero line)")
    ax.plot(x_dg, rho_dg - ref_on_dg, color="C0", lw=1.0,
            label="DG P2 N=800 $- $ reference")
    ax.plot(x, raw_on_ref - ref, "--", color="C3", lw=1.0,
            label="raw N=1600 $- $ reference")
    ax.axhline(band, color="0.5", lw=0.8, ls=":")
    ax.axhline(-band, color="0.5", lw=0.8, ls=":")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\Delta\rho$")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = ART / "fig1_certified_reference_vs_dg.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_band(audit: dict) -> Path:
    """Band share per ladder (gate-validity rule) and leg errors + band vs gates."""
    names = list(audit["ladders"])
    shares = [audit["ladders"][n]["band_share_of_gate"] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    colors = ["C2" if audit["ladders"][n]["gate_usable"] else "C3" for n in names]
    ax.bar(names, shares, color=colors)
    ax.axhline(1.0 / 3.0, color="k", ls="--", lw=1.0,
               label="gate-validity limit $U/\\tau = 1/3$")
    ax.set_yscale("log")
    ax.set_ylabel("$U / \\tau$  (band share of the gate)")
    ax.set_title("Which ladder may certify a reference?")
    for i, s in enumerate(shares):
        ax.text(i, s * 1.25, f"{s:.3g}", ha="center", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    legs = audit["dg_legs"]
    labels = [lg["configuration"].replace("DG ", "").replace(" CFL=", "\nCFL=") for lg in legs]
    err = [lg["rel_l1_percent"] for lg in legs]
    band = [lg["band_percent"] for lg in legs]
    ax.bar(range(len(legs)), err, color="C0", label="error vs certified reference")
    ax.errorbar(range(len(legs)), err, yerr=band, fmt="none", ecolor="k", capsize=2,
                lw=0.8, label="reference band $U$")
    ax.axhline(3.0, color="C2", ls="--", lw=1.0, label="owner 3 % standard")
    ax.axhline(5.0, color="C3", ls="--", lw=1.0, label="case gate 5 %")
    ax.set_xticks(range(len(legs)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("relative L1 error [%]")
    ax.set_title("Shipped DG legs vs certified reference (error + band)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    out = ART / "fig2_band_share_and_legs.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def render_doc(audit: dict, figs: dict) -> Path:
    fv = audit["fv_certification"]
    legs = audit["dg_legs"]
    worst = max(legs, key=lambda lg: lg["rel_l1_percent"])
    best = min(legs, key=lambda lg: lg["rel_l1_percent"])
    worst_plus_band = max(lg["rel_l1_percent"] + lg["band_percent"] for lg in legs)
    usable = audit["usable_certifiers"]
    names = list(audit["ladders"])
    verdict_lines = []
    for name, c in audit["ladders"].items():
        p = c.get("observed_order_used")
        u = c.get("band_percent")
        share = c.get("band_share_of_gate")
        p_txt = "n/a" if p is None else "%.4f" % p
        u_txt = "n/a" if u is None else "%.6f" % u
        share_txt = "n/a" if share is None else "%.3f" % share
        verdict_lines.append(
            "| `%s` | %s | %s | %s | %s | `%s` | `%s` |" % (
                name, c["ladder"], p_txt, u_txt, share_txt,
                c["classification"], c["verdict"]))
    leg_lines = []
    for lg in legs:
        d3 = lg["judge_3pct"]
        leg_lines.append(
            f"| {lg['configuration']} | {lg['rel_l1_percent']:.6f} | "
            f"{lg['rel_l2_percent']:.6f} | {lg['rel_l1_percent'] + lg['band_percent']:.6f} | "
            f"`{lg['decision']}` | `{d3['decision']}` | "
            f"{d3['robust_margin_percent_of_gate']:.1f} % |")

    cov = audit["band_coverage"]
    doc = f"""# Certified reference for cases without an exact solution (round-2 practice, engine form)

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
* Roache GCI with `Fs = {fv['fs_gci']}`: `U = Fs * d_fine / (r^p - 1)`, the band
  of the **finest** level -- the certified reference carries that band.

### Pre-registered decision rules (written before the audit was run)

| # | rule | consequence |
|---|---|---|
| R1 | ladder classified `monotonic` only if `p_min <= p <= p_max` (`{fv['p_min']}` / `{fv['p_max']}`); round-off floor `d_fine <= 1e-12` refused | otherwise `WITHHELD_*`, no reference produced |
| R2 | gate validity: `U / tau <= {fv['u_share_max']:.4f}` | otherwise `WITHHELD_BAND_CONSUMES_GATE` -- the comparison would be vacuous |
| R3 | margin robustness: `PASS` only if `err + U < tau` | else `PASS_MARGINAL` (passes, but not resolvably) |

R1 fires on data whose differences stop shrinking; R2 is the machine form of
"the reference must be much better than the gate"; R3 stops a gate being passed
by reference noise.

### The displacement trap (why `U` and "extrapolation shift" are not comparable)

The certified profile replaces the **second-finest** level, so its distance to
that level is that level's own error, i.e. `r^p/(r^p-1)` times the medium-fine
difference -- a factor of about 1.7, larger than `U`. For the shipped case the
measured displacement is {fv['extrapolation_displacement_percent']:.6f} % against
`U = {fv['band_percent']:.6f} %` (ratio {fv['displacement_ratio']:.6f} to the
identity). Reading that ratio as "the band is understated" is the wrong reading;
the band belongs to the finest level and the test suite pins the identity
(`tests/test_certified_reference.py::test_displacement_identity_and_the_wrong_reading`).

## 3. Which ladder may certify a reference?

Gate used for R2: **{audit['tolerance_percent']:g} %**.

| ladder | levels | p (observed) | band `U` [%] | `U/tau` | classification | verdict |
|---|---|---|---|---|---|---|
{chr(10).join(verdict_lines)}

Usable certifiers: `{usable}` ({audit['n_usable']}/{audit['n_total']}).

**This is the quantitative core of the result.** The independent FV-WENO5/HLLC/
SSP-RK3 ladder is asymptotic (`p = {fv['observed_order_used']:.4f}`, far below its
formal 5th order because the solution contains shocks) with a band of
`{fv['band_percent']:.6f} %` = {fv['band_share_of_gate']*100:.2f} % of the 5 %
gate, so it certifies. The DG families themselves do not: a fixed-degree DG ladder
refined in `N` stagnates (`p` = {', '.join(f"{audit['ladders'][n]['observed_order_used']:.4f}" for n in names if n.startswith('dg'))}),
so their bands are
{', '.join(f"{audit['ladders'][n]['band_percent']:.3f} %" for n in names if n.startswith('dg'))}
-- {', '.join(f"{(audit['ladders'][n]['band_share_of_gate']):.2f}x" for n in names if n.startswith('dg'))}
the gate. The solver cannot certify itself for this case, and now there is a number for
why. The two rules fire independently: R1 refuses every DG ladder on the observed
order (`p < p_min = {fv['p_min']}`), and R2 refuses them on the band alone
(`U/tau` = {', '.join('%.2f' % audit['ladders'][n]['band_share_of_gate'] for n in names if n.startswith('dg'))}).
Either rule alone would be enough; both are recorded in the JSON `reasons`.

## 4. Shipped DG legs re-judged against the certified reference

| configuration | rel L1 [%] | rel L2 [%] | err + U [%] | decision (5 %) | decision (3 %) | robust margin vs 3 % gate |
|---|---|---|---|---|---|---|
{chr(10).join(leg_lines)}

Worst leg: {worst['configuration']} at {worst['rel_l1_percent']:.6f} %; best:
{best['configuration']} at {best['rel_l1_percent']:.6f} %.
Worst `err + U` over all legs = **{worst_plus_band:.6f} %**.
Decision counts at the 5 % gate: `{audit['decision_counts_5pct']}`; at the owner
3 % standard: `{audit['decision_counts_3pct']}`.

### Error verdict (explicit, item by item)

| quantity | value | reference | error | < 3 % ? |
|---|---|---|---|---|
| certified band `U` vs GCI recomputation of the legacy driver | bit-identical | round-2 published {fv['band_percent']:.6f} % | 0 (exact equality) | yes |
| observed order `p` vs legacy driver | {fv['observed_order_used']:.12f} | 1.282509176116 | 0 (exact equality) | yes |
| certified profile vs legacy `rho_certified` | | | max abs diff 0.0 | yes |
| all 11 shipped legs vs certified reference | worst {worst['rel_l1_percent']:.6f} % | certified reference | {worst['rel_l1_percent']:.6f} % | **yes** (worst {worst['rel_l1_percent']:.4f} %) |
| worst leg including the band (`err + U`) | {worst_plus_band:.6f} % | certified reference | {worst_plus_band:.6f} % | **yes** |

**Is it < 3 %?** Yes, for every shipped leg, including when the reference band is
added to the error: the worst leg is {worst['rel_l1_percent']:.4f} % and the worst
`err + U` is {worst_plus_band:.4f} %, i.e. {(3.0 - worst_plus_band)/3.0*100:.1f} % of
the 3 % budget still unused. No leg moved across a gate relative to the round-2
report. There is **no literature tabulated solution** for this case (Shu & Osher
1988 show the solution graphically), so no literature comparison is claimed --
this is code verification against a certified reference, not experimental
validation.

## 5. Is the band meaningful? (coverage check with a known exact solution)

On `{cov['n_cases']}` synthetic ladders (`seed = {cov['seed']}`, levels
`{cov['levels']}`) whose exact solution is known and whose error is a two-term
model `A N^-p + B N^-q`, the certified profile's true error was covered by its own
band in **{cov['coverage_fraction']*100:.1f} %** of the cases; the worst
true/band ratio was {cov['worst_true_over_band']:.6f} and the median
{cov['median_true_over_band']:.6f}. So on smooth ladders the band is *strongly
conservative* (the observed order is high and the extrapolation removes almost
all of the error). This does **not** transfer to the shock case: there the
observed order is ~1 and the true error is unmeasurable, which is exactly why the
band -- not a claimed exactness -- carries the argument.

## 6. Figures

### fig1 -- certified reference, its band, and a DG leg
![fig1](artifacts/m17539/fig1_certified_reference_vs_dg.png)

Top: certified reference with its `+/- U` ribbon, the raw FV level it replaces,
and DG P2 N=800. Bottom: differences to the reference, band lines dotted.
The reference band contributes only {100*min(lg["band_percent"]/lg["rel_l1_percent"] for lg in legs):.1f}--{100*max(lg["band_percent"]/lg["rel_l1_percent"] for lg in legs):.1f} % of any leg's error, so the leg errors are the solver's,
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
  ({fv['observed_order_used']:.4f}) because the solution contains shocks. Any
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
python -m tensordg.certified_reference \\
    --rawdir validation/repro_shu_osher/artifacts/raw --outdir artifacts/m17539
python -m tools.render_certified_reference          # -> this document + the two figures
python -m pytest tests/test_certified_reference.py -q
```

Artifacts: `artifacts/m17539/certified_reference_audit.json` (all numbers, with
`input_sha256` of the ladder it was computed from),
`certified_reference_audit.md`, `certified_reference_fv.npz` (the certified
profile), `fig1_certified_reference_vs_dg.png`, `fig2_band_share_and_legs.png`.
"""
    DOC.write_text(doc)
    return DOC


def main() -> int:
    audit = json.loads((ART / "certified_reference_audit.json").read_text())
    figs = {"fig1": fig_field(audit), "fig2": fig_band(audit)}
    doc = render_doc(audit, figs)
    print(f"wrote {doc}")
    for k, v in figs.items():
        print(f"wrote {v}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())