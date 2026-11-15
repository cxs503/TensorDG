#!/usr/bin/env bash
# Steer for the embedded-wall vs boundary-wall parity ladder (gate G3 of
# REPRO-DG-CKS1998); driver: benchmarks/cks1998_embedded_wall_parity.py.
#
# Every arm is an independent single-threaded python process and the per-step
# cost is python overhead rather than arithmetic, so the arms are launched in
# parallel instead of being marched one after another.
set -u
export OMP_NUM_THREADS=1
export PYTHONPATH=.
OUT=${OUT:-docs/benchmarks/results/cks1998-embedded-wall}
RES=${RES:-"60 120 240"}
DEG=${DEG:-"1 2"}
JOBS=${JOBS:-12}
mkdir -p "$OUT"

python -m benchmarks.cks1998_embedded_wall_parity \
  --resolutions $RES --degrees $DEG --plan \
  | xargs -P "$JOBS" -I{} python -u -m benchmarks.cks1998_embedded_wall_parity \
      --single {} --outdir "$OUT"
rc=$?
python -m benchmarks.cks1998_embedded_wall_parity --aggregate --outdir "$OUT" || exit 1
exit $rc