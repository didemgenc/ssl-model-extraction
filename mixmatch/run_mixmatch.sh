#!/bin/bash
# MixMatch baseline on our victim/query/test data, one run per budget (resumes automatically).
# Usage: [SETUP=original|split] bash run_mixmatch.sh [epochs] [budgets...]   default: original, 128 epochs, 4k 1k 250
cd "$(dirname "$0")"
EP=${1:-128}; shift; BUDGETS=${@:-4k 1k 250}; SETUP=${SETUP:-original}
mkdir -p outputs
for b in $BUDGETS; do
  if [ -f outputs/mixmatch_results.csv ] && grep -q "^$SETUP,$b,$EP," outputs/mixmatch_results.csv; then echo "skip $b (done)"; continue; fi
  echo "=== mixmatch setup $SETUP budget $b epochs $EP $(date)" | tee -a outputs/mixmatch_run.log
  python train_mixmatch_mea.py --setup $SETUP --budget $b --epochs $EP 2>&1 | grep -E "Labeled|resumed|epoch|RESULT|Error|error" | tee -a outputs/mixmatch_run.log
done
echo "DONE $(date)" | tee -a outputs/mixmatch_run.log
