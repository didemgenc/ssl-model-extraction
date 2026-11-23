#!/bin/bash
# Pause the training process when the CPU package gets too hot, resume when it cools down.
# Usage: bash thermal_guard.sh <pid-pattern>  e.g. "python table2_proposed"
# Pauses at >= $HOT C, resumes at <= $COOL C (coretemp package sensor).
PAT=${1:?pattern}
HOT=${HOT:-99}; COOL=${COOL:-92}
SENSOR=$(grep -l coretemp /sys/class/hwmon/hwmon*/name | head -1 | xargs dirname)/temp1_input
paused=0
while true; do
  t=$(( $(cat $SENSOR) / 1000 ))
  pids=$(pgrep -f "^$PAT")
  if [ -n "$pids" ]; then
    if [ $paused -eq 0 ] && [ $t -ge $HOT ]; then kill -STOP $pids; paused=1; echo "$(date +%T) ${t}C pause"; fi
    if [ $paused -eq 1 ] && [ $t -le $COOL ]; then kill -CONT $pids; paused=0; echo "$(date +%T) ${t}C resume"; fi
  fi
  sleep 5
done
