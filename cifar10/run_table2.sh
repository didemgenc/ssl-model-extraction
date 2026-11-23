#!/bin/bash
# Proposed-method row of Table 2 (CIFAR-10), paper thresholds, one run per budget per seed.
# Usage: bash run_table2.sh [seed ...]   (default 950). Runs already in outputs/table2_results.csv are skipped,
# so the script can simply be re-launched after an interruption.
cd "$(dirname "$0")"
SEEDS=${@:-950}
LOG=outputs/table2_run.log
CSV=outputs/table2_results.csv
mkdir -p outputs
for seed in $SEEDS; do
  for budget in 4k 2k 1k 250; do
    if [ -f $CSV ] && grep -qE "^$budget,[^,]*,[^,]*,$seed," $CSV; then
      echo "skip seed $seed budget $budget (done)" | tee -a $LOG; continue
    fi
    echo "=== seed $seed budget $budget  $(date)" | tee -a $LOG
    python table2_proposed.py --budget $budget --seed $seed 2>&1 | grep -E 'budget|RESULT|Error|error' | tee -a $LOG
  done
done
echo "DONE $(date)" | tee -a $LOG
