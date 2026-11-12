#!/usr/bin/env bash
# One-command reproduction of the Shu--Osher (1988) shock/entropy-wave DG benchmark.
#
#   bash validation/repro_shu_osher/run_all.sh          # full study  (~30 min, 8 procs)
#   JOBS=1 bash validation/repro_shu_osher/run_all.sh   # serial
#
# Every configuration is memoised in runs/shu_osher_case_pool/raw/<key>.npz, so an
# interrupted run resumes instead of recomputing.  Threads are pinned to one per
# process on purpose: the DG kernel is small, and letting torch/BLAS spawn one
# thread per core on a shared 128-core host is 5-10x slower than single-threaded
# (measured: 120 s vs 25 s for the N=200 P1 leg).
set -u
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
JOBS=${JOBS:-8}

# independent reference solver: cell-centred FV-WENO5 / HLLC / SSP-RK3
printf '%s\n' 800 1600 3200 | xargs -P "$JOBS" -n 1 \
  python -m benchmarks.shu_osher_case_pool reference --ncells

# DG ladder: P1/P2/P3 x N = 200/400/800 (WENO limiter, CFL = 0.4)
printf '%s\n' "1 200" "1 400" "1 800" "2 200" "2 400" "2 800" "3 200" "3 400" "3 800" \
  | xargs -P "$JOBS" -n 2 sh -c 'python -m benchmarks.shu_osher_case_pool ladder --degree "$0" --ncells "$1"'

# CFL sensitivity at N = 400, P2 (CFL = 0.4 comes from the ladder above)
printf '%s\n' 0.1 0.2 | xargs -P "$JOBS" -n 1 sh -c \
  'python -m benchmarks.shu_osher_case_pool ladder --degree 2 --ncells 400 --cfl "$0"'

# merge -> runs/shu_osher_case_pool/metrics.json + profiles.npz
python -m benchmarks.shu_osher_case_pool merge --reference-cells 3200

# one fresh configuration end-to-end (proves the artifacts come from the code)
python validation/repro_shu_osher/smoke_repro.py

# publish the curated bundle: benchmark_ref_vs_got.{json,md}, evidence.md,
# artifacts/ (incl. sha256 manifest) read back from runs/
python validation/repro_shu_osher/make_evidence.py