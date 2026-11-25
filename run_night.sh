#!/bin/bash
# Overnight queue: MixMatch (Table 3 setting, single seed) for 4k 1k 250, then the proposed method in the same
# setting with a single seed (950), ~40 min. Both parts resume/skip finished work, so the script can simply be re-run after a crash.
# Usage: bash run_night.sh [mixmatch_epochs]      (default 32)
cd "$(dirname "$0")"
EP=${1:-32}
nohup bash thermal_guard.sh "python" > thermal_guard.out 2>&1 &
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
nice -n 10 bash mixmatch/run_mixmatch.sh $EP 4k 1k 250
nice -n 10 bash cifar10/run_table3.sh 950
echo "NIGHT DONE $(date)"
