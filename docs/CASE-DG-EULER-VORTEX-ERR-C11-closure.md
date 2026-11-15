# CASE-DG-EULER-VORTEX-ERR: C11 Provenance Closure Plan

> Anchor: `CASE-DG-EULER-VORTEX-ERR` | Mission: m22487
> Date: 2026-11-15

## Problem Statement

The C11 (provenance integrity) gate was failing because 5 legacy result records
lacked replay certificates. These records were produced before the
`reference_model` field was added to the result schema, so their provenance
was unknown.

### Unverified Records

| Task Key | Config | Estimated Wall Time |
|---|---|---|
| `dg_d2_n32_t5_cfl0.2` | P2, 32², t=5, CFL=0.2 | ~150 s |
| `dg_d2_n32_t10_cfl0.2` | P2, 32², t=10, CFL=0.2 | ~300 s |
| `dg_d2_n32_t10_cfl0.4` | P2, 32², t=10, CFL=0.4 | ~210 s |
| `dg_d2_n64_t2_cfl0.2` | P2, 64², t=2, CFL=0.2 | ~340 s |
| `dg_d2_n64_t10_cfl0.2` | P2, 64², t=10, CFL=0.2 | ~1700 s |

## Resolution

### Cluster Job

Submitted cluster job `cj4164` to run:
```bash
PYTHONPATH=. python3 tools/provenance_certify.py --remaining --max-seconds 3600 --threads 4
```

This will replay all 5 unverified checkpoints against the current `min_image`
reference and generate replay certificates.

### New Test Added

Added `test_c11_all_legacy_records_have_certificates` to
`tests/test_case_euler_vortex_errconserve.py`. This test:
1. Scans all result files in `case_euler_vortex_errconserve/results/`
2. Identifies legacy records (no `reference_model` field)
3. Checks that each has a corresponding replay certificate
4. Fails with a clear message listing unverified records and the command to fix them

### Expected Outcome After Job Completion

Once `cj4164` completes:
1. Re-run aggregation: `PYTHONPATH=. python3 benchmarks/case_euler_vortex_errconserve.py --aggregate --figures --output-dir case_euler_vortex_errconserve`
2. C11 gate will change from "未通过" to "通过"
3. `provenance_certificates.json` will have 19 certificates (14 existing + 5 new)
4. All 14 existing tests + 1 new test will pass

## Current Gate Status (Pre-Closure)

| Criterion | Status | Notes |
|---|---|---|
| C1 (t=1 tolerance < 6%) | ✅ PASS | worst 4.9260% (p0_n64) |
| C2 (error conservation alpha ≤ 1) | ✅ PASS | alpha = 0.313 (dg_d2_n32) |
| C3 (discrete conservation < 1e-10) | ✅ PASS | worst 3.911e-13 |
| C4 (entropy budget < 1e-3) | ✅ PASS | 6.336e-07 |
| C5 (cross-check order ≥ 0.8) | ✅ PASS | 0.849 |
| C6 (ladder completeness) | ✅ PASS | 40/40 |
| C7 (design space invariance) | ✅ PASS | max alpha 0.440 |
| C8 (degree invariance) | ✅ PASS | max alpha 0.440 |
| C9 (design order) | ✅ PASS | worst 2.042 |
| C10 (direction invariance) | ✅ PASS | max alpha 0.313 |
| C11 (provenance integrity) | ⏳ PENDING | 5 unverified → cluster job cj4164 |

## Key Physical Findings

The isentropic vortex transport case demonstrates:

1. **Error Conservation**: For high-order DG (P1-P3), the error is a mesh property,
   not a time property. The cumulative exponent alpha ranges from 0.025 (P3) to
   0.440 (P1), all well below the alpha ≤ 1 bound.

2. **Order Convergence**: Spatial order at t=10 reaches 2.042 (P1), 4.639 (P2),
   confirming the theoretical p+1 convergence.

3. **Discrete Conservation**: Mass/momentum/energy drift at round-off level
   (~1e-13), confirming the telescoping flux property of periodic DG.

4. **Entropy Budget**: Entropy drift ~6e-07, consistent with the isentropic
   nature of the exact solution and the low numerical dissipation of HLLC DG.

5. **P0 vs DG Contrast**: The first-order P0 scheme shows significant error
   accumulation (alpha = 0.479 for n=64, with absolute error growing from
   4.9% to 14.3% over t=1→10), while DG-P2 error grows only from 0.0094% to
   0.0152% over the same interval.

## Reproduction

```bash
# Run the full ladder (sharded for parallelism)
PYTHONPATH=. python3 benchmarks/case_euler_vortex_errconserve.py --shard i/n --output-dir case_euler_vortex_errconserve

# Aggregate and generate figures
PYTHONPATH=. python3 benchmarks/case_euler_vortex_errconserve.py --aggregate --figures --output-dir case_euler_vortex_errconserve

# Generate provenance certificates for legacy records
PYTHONPATH=. python3 tools/provenance_certify.py --remaining --max-seconds 3600 --threads 4

# Run tests
pytest tests/test_case_euler_vortex_errconserve.py -v
pytest tests/test_euler_vortex_error_conservation.py -v
```
