#!/usr/bin/env bash
# One-key reproduction of CASE-DG-EULER-VORTEX-ERR (isentropic vortex, one period).
#
#   bash tools/run_case_yee_vortex.sh            # default: 4 parallel configurations
#   JOBS=1 bash tools/run_case_yee_vortex.sh     # serial (single-processor node)
#   bash tools/run_case_yee_vortex.sh aggregate  # only rebuild tables/figures/report
#
# Every configuration is an independent process writing its own checkpoint JSON, so
# an interrupted run only loses the configurations that were still in flight.
# Heavy (>10 min) runs belong on the scheduler: submit this same command through the
# compute gate instead of running it on the control plane.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
OUT="${OUT:-runs/case_yee2000_vortex}"
JOBS="${JOBS:-4}"

n=$(python3 benchmarks/case_euler_vortex_yee.py --task list | wc -l)

if [[ "${1:-run}" == "aggregate" ]]; then
  python3 benchmarks/case_euler_vortex_yee.py --task aggregate --output-dir "${OUT}"
  exit 0
fi

python3 benchmarks/case_euler_vortex_yee.py --task residual --output-dir "${OUT}"
seq 0 $((n - 1)) | xargs -P "${JOBS}" -I{} \
  python3 benchmarks/case_euler_vortex_yee.py --config-index {} --output-dir "${OUT}" --threads 1

python3 benchmarks/ref_provenance_yee2000.py || true
python3 benchmarks/case_euler_vortex_yee.py --task aggregate --output-dir "${OUT}"