#!/bin/bash
# Reproduce all 24 cells of Table 1 (4 budgets x 2 heads x 3 training methods).
# Usage: bash run_table1.sh [seed ...]   (default seed 950 = the one used in the original script)
cd "$(dirname "$0")"
SEEDS=${@:-950}
LOG=outputs/table1_run.log
CSV=outputs/table1_results.csv
mkdir -p outputs
for seed in $SEEDS; do
  for budget in 4k 2k 1k 250; do
    for method in frozen layer4 finetune; do
      for model in linear MLP; do
        if [ -f $CSV ] && grep -qE "^$budget,$model,$method,$seed," $CSV; then echo "skip $budget $model $method seed $seed (done)"; continue; fi
        echo "=== seed $seed budget $budget method $method model $model  $(date)" | tee -a $LOG
        python table1_baseline.py --budget $budget --training_method $method --model $model --seed $seed 2>&1 | grep -E 'budget|RESULT|Error|error' | tee -a $LOG
      done
    done
  done
done
echo "DONE $(date)" | tee -a $LOG
