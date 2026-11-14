#!/bin/bash
# Launch the CASE-EULER-VORTEX study ladder as independent, detached
# single-threaded tasks (one process per configuration -> one checkpoint file).
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=${1:-case_euler_vortex}
mkdir -p "$OUT/results" "$OUT/logs"
export OMP_NUM_THREADS=1

launch() {
  # `-m` keeps the repo root on sys.path: a bare
  # `python3 benchmarks/case_euler_vortex.py` cannot import the tensordg
  # package from a plain checkout (ModuleNotFoundError).
  local key="$1"; shift
  setsid --fork python3 -u -m benchmarks.case_euler_vortex --output-dir "$OUT" \
      --threads 1 "$@" > "$OUT/logs/$key.log" 2>&1 < /dev/null
}

launch exact_solution_residual --task residual
for n in 32 64 96 128; do
  launch "p0_n${n}_t1_cfl0.4" --task p0 --cells "$n" --final-time 1.0 --cfl 0.4
done
for n in 16 32 64; do
  launch "dg_d1_n${n}_t1_cfl0.2" --task dg --degree 1 --cells "$n" --final-time 1.0 --cfl 0.2
done
for n in 16 32 48; do
  launch "dg_d2_n${n}_t1_cfl0.2" --task dg --degree 2 --cells "$n" --final-time 1.0 --cfl 0.2 \
      $( [ "$n" = 32 ] && echo --save-fields )
done
for n in 12 16 24; do
  launch "dg_d3_n${n}_t1_cfl0.2" --task dg --degree 3 --cells "$n" --final-time 1.0 --cfl 0.2
done
# sensitivity: CFL at the gate configuration, then final time
launch "dg_d2_n32_t1_cfl0.1" --task dg --degree 2 --cells 32 --final-time 1.0 --cfl 0.1
launch "dg_d2_n32_t1_cfl0.4" --task dg --degree 2 --cells 32 --final-time 1.0 --cfl 0.4
launch "dg_d2_n32_t0.5_cfl0.2" --task dg --degree 2 --cells 32 --final-time 0.5 --cfl 0.2
launch "dg_d2_n32_t2_cfl0.2" --task dg --degree 2 --cells 32 --final-time 2 --cfl 0.2
echo "launched $(ls "$OUT/results" | wc -l) done so far; jobs running in background"