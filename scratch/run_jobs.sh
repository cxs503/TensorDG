#!/bin/bash
cd /root/tp-agent/work/m16075
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KOVASZNAY_THREADS=1 PYTHONPATH=.
for name in "$@"; do
  echo "=== $name start $(date +%H:%M:%S)"
  python3 -m benchmarks.verify_kovasznay_dg_2d --job "$name" >> scratch/job_${name}.log 2>&1
  echo "=== $name rc=$? $(date +%H:%M:%S)"
done
