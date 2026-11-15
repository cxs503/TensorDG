#!/usr/bin/env bash
# Run the 8 missing ladder items for CASE-DG-EULER-VORTEX-ERR
# Submitted by m21734 to complete C6 (ladder completeness) and C9 (design order)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="."
export OMP_NUM_THREADS=1

OUTDIR="case_euler_vortex_errconserve"
mkdir -p "$OUTDIR/results"

# Missing items:
# dg_d3_n32: t=1,2,5,10 (P3 degree, 32 cells, CFL=0.2)
# dg_d1_n64: t=2,5,10 (P1 degree, 64 cells, CFL=0.2) -- t=1 already exists
# dg_d2_n32_t20: (P2 degree, 32 cells, CFL=0.2, t=20)

run_one() {
    local task="$1" degree="$2" cells="$3" ft="$4" cfl="$5"
    local key="${task}_d${degree}_n${cells}_t${ft//./}_cfl${cfl//./}"
    # Handle direction suffix if needed
    if [ "$task" = "dg" ] && [ "$degree" = "2" ] && [ "$cells" = "32" ] && [ "$ft" = "20.0" ]; then
        key="dg_d2_n32_t20_cfl0.2"
    fi
    local outfile="$OUTDIR/results/${key}.json"
    if [ -f "$outfile" ]; then
        echo "SKIP existing $outfile"
        return 0
    fi
    echo "RUNNING $key ..."
    python3 benchmarks/case_euler_vortex_errconserve.py \
        --task "$task" --degree "$degree" --cells "$cells" \
        --final-time "$ft" --cfl "$cfl" \
        --output-dir "$OUTDIR" --threads 1
}

# P3 degree, 32 cells, CFL=0.2
run_one dg 3 32 1.0 0.2
run_one dg 3 32 2.0 0.2
run_one dg 3 32 5.0 0.2
run_one dg 3 32 10.0 0.2

# P1 degree, 64 cells, CFL=0.2 (t=2,5,10; t=1 exists)
run_one dg 1 64 2.0 0.2
run_one dg 1 64 5.0 0.2
run_one dg 1 64 10.0 0.2

# P2 degree, 32 cells, CFL=0.2, t=20
run_one dg 2 32 20.0 0.2

echo "All missing ladder items completed."
