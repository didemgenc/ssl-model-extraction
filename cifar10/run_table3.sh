#!/bin/bash
# Proposed method in the Table 3 / MixMatch setting. Usage: bash run_table3.sh [seeds...]  (budgets 4k 1k 250)
cd "$(dirname "$0")"
SEEDS=${@:-950 1 2}; CSV=outputs/table3_results.csv; BUDGETS=${BUDGETS:-4k 1k 250}
for seed in $SEEDS; do for budget in $BUDGETS; do
  if [ -f $CSV ] && grep -qE "^$budget,[^,]*,[^,]*,$seed," $CSV; then echo "skip $budget seed $seed (done)"; continue; fi
  echo "=== table3 budget $budget seed $seed $(date)" | tee -a outputs/table3_run.log
  python table3_proposed.py --budget $budget --seed $seed 2>&1 | grep -E "victim|training samples|RESULT|Error|error" | tee -a outputs/table3_run.log
done; done
echo "DONE $(date)" | tee -a outputs/table3_run.log
