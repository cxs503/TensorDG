#!/bin/bash
cd /root/tp-agent/work/m16075
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KOVASZNAY_THREADS=1 PYTHONPATH=.
deadline=$(( $(date +%s) + ${1:-1500} ))
for f in primary p1_12 p2_12 mach030; do
  while [ ! -f validation/kovasznay/$f.json ] && [ $(date +%s) -lt $deadline ]; do sleep 10; done
done
for f in wavenumber3 bc_extrap p1_24 p2_24; do
  [ -f validation/kovasznay/$f.json ] || sleep 0
done
python3 -m benchmarks.verify_kovasznay_dg_2d --job summary > scratch/final_summary.log 2>&1
python3 -m tools.gen_case_kovasznay_report > scratch/final_doc.log 2>&1
python3 publish_case_kovasznay_report.py > scratch/final_publish.log 2>&1
{ echo "DONE $(date +%H:%M)"; ls validation/kovasznay/; echo ---; tail -3 scratch/final_publish.log; } > scratch/FINALIZE_DONE
