# KPI-TOOLS-5: external tool integration -- TensorDG solver-side probes

**Anchor**: KPI-TOOLS-5 (tool classes >= 5)  
**Verdict**: `PASS` -- real_classes=6/6  
**Generated**: 2026-11-14T07:16:10+00:00 (numbers injected from `run_all()`)

## Tool class x DG surface x reference

| tool class | module | version | metric | observed | reference | abs err | limit | verdict |
|---|---|---|---|---|---|---|---|---|
| autodiff | torch | 2.7.1 | max(rel err grad(2Mu), rel err dE/dt) | -9.879168e+00 | -9.879168e+00 | 5.785406e-18 | 1.000000e-12 | REAL |
| numerical_integration | scipy | 1.16.0 | rel diff(adaptive quad, DG Gauss) of L2 error (+anchor) | 1.069582e-04 | 1.069582e-04 | 2.265551e-13 | 1.000000e-08 | REAL |
| symbolic_math | sympy | 1.14.0 | max abs err float64 DG volume matrix vs exact rationals | 9.325873e-15 | 0 | 9.325873e-15 | 1.000000e-13 | REAL |
| arbitrary_precision | mpmath | 1.3.0 | abs err 40-digit int_0^1 x^2 sin^2(pi x) vs 1/6 - 1/(4 pi^2) | 1.413364e-01 | 1.413364e-01 | 2.869859e-42 | 1.000000e-38 | REAL |
| spectral_analysis | numpy.fft | 2.4.6 | max(|c1| - 1, higher-mode leakage fraction) | 9.999996e-01 | 1.000000e+00 | 3.586565e-07 | 1.000000e-06 | REAL |
| visualization | matplotlib | 3.10.8 | |y-span - exact peak-to-peak 2.0| (PNG payload > 1 kB required) | 2.000012e+00 | 2.000000e+00 | 1.211067e-05 | 1.000000e-03 | REAL |

## Probe details

- **autodiff** (REAL): heat_1d SIPG ncells=8 degree=2 dofs=24; rel err grad=5.785e-18, rel err dE/dt=0.000e+00
- **numerical_integration** (REAL): anchor int_0^1 sin^2 = 0.49999999999999994 (err 5.55e-17); DG P2 ncells=20 t=0.5 steps=625: quad L2=1.069581548119e-04, solver Gauss L2=1.069581548119e-04, rel diff=2.266e-13
- **symbolic_math** (REAL): degree=3: exact V rows=[[0, 0, 0, 0], [2, 0, 0, 0], [0, 2, 0, 0], [2, 0, 2, 0]]; mass identity residual exact=True
- **arbitrary_precision** (REAL): dps=40 err=2.870e-42; heat energy ref (t=0.1) hp err=0.000e+00, float64 vs hp rel diff=1.998e-16 (= float64 floor of the DG heat reference)
- **spectral_analysis** (REAL): DG P2 ncells=32 t=1.0 steps=2000: |c1|=0.999999641343 (err 3.59e-07), leakage=1.189e-09
- **visualization** (REAL): png=13529 B, y-span=2.000012 (payload_ok=True), max|u_h - exact|=6.446e-05, ncells=40 h=0.0250

## Scope and limits

- These probes establish *tool usability on genuine DG surfaces*, not new
  solver accuracy: every probe's reference is closed form or exact rational
  arithmetic, so the scored error is the tool+operator chain, not a physics model.
- A missing tool reports `MISSING` and forces `verdict = FAIL`; nothing is
  silently skipped.

## Evidence figure

![external tools](artifacts/external_tools/external_tools_dg.png)

Left: the DG P2 field the `matplotlib` probe renders against the exact wave
(the same run scores the visualization class). Right: every tool class's
`abs_err / limit` ratio on a log axis -- all bars sit far left of the red
acceptance threshold, i.e. the tool chains are exact to round-off.

## Reproduce

```bash
PYTHONPATH=. python3 -m pytest tests/test_external_tools_verification.py -q
PYTHONPATH=. python3 tools/build_external_tools_dg_report.py
```

## Independence limitations (adversarial self-review)

- The probes run in-process against TensorDG's own operators; the references
  (exact rationals for sympy, closed forms for the integrals, the analytic
  sine wave for fft/matplotlib) are independent of the DG code, but the
  operator *assembly* being exercised is the one under test. A defect shared
  by assembly and probe (e.g. the same Legendre normalisation error) would
  not be caught here; it is caught by the convergence suites in `tests/`.
- `version` is the installed module version, not a hash; minimum versions in
  `PROBE_SPEC` are declared intent, not enforced by an environment lock.
- The probes are CPU/float64 only; GPU (`cuda`) and lower-precision dtypes are
  out of scope.
- These are tool-availability-plus-correctness probes, not new physics
  validation: no benchmark verdict (e.g. Thacker, Sod) is claimed or changed.

## What this evidence does *not* establish

- It does not claim the platform-level KPI is closed on its own: it is the
  TensorDG-side contribution (6 classes on genuine DG surfaces). The KPI
  counts classes across the ecosystem, so the ledger evidence has to be
  updated with this record alongside the other repositories' records.
- It does not verify tool behaviour under parallelism/MPI or on the loong64
  cluster nodes.
