# CASE-LAX-DG · resolution requirement of the ledger acceptance bar

> Generated from `metrics.json` by `python -m benchmarks.lax_resolution_requirement`
> (measurement run 2026-11-14T18:39:16Z, wall 246.2 s).
> Ledger bar: relative error vs exact Riemann solution < 3%.

## Question

`p3.N400` (the pinned A0 acceptance leg) misses the bar:
worst whole-field relative L2 `3.11556e-02`
(`density`, margin x1.039 of
the tolerance). Does one further h-refinement reach it?

Rule (pre-registered): resolution requirement = the smallest leg of the extension ladder whose whole-field relative L2 is below the registered tolerance for every field (no field exemption, no wave band removed, no averaging); a leg is PASS iff max_field rel L2 < tolerance.

## Ledger metric per leg

| leg | worst rel L2 | field | tolerance margin (x) | status |
|---|---|---|---|---|
| p3.N400 | 3.11556e-02 | density | 1.039 | FAIL |
| p3.N800 | 2.31490e-02 | density | 0.772 | PASS |

Smallest leg meeting the bar: **p3.N800**
(resolution factor x2.0);
R2 verdict **PASS**.

## Measured order, N=400 -> N=800

| field | order (L1) | order (L2) |
|---|---|---|
| density | 0.495 | 0.429 |
| velocity | 0.580 | 0.701 |
| pressure | 0.511 | 0.537 |

## Reuse check

Committed artifacts re-measured here: worst relative difference
`7.138e-03` (tolerance
`1e-06`), status
**FAIL**.

## Scope

A0 stays pinned at `p3.N400`: this artifact adds a resolution
statement, it does not relabel the acceptance verdict.
