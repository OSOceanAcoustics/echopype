#!/usr/bin/env bash

mkdir -p ci_monitor
echo "timestamp,mem_used_mb" > ci_monitor/memory_usage.csv

while true; do
  ts=$(date +%s)
  used=$(free -m | awk '/^Mem:/ {print $3}')
  echo "$ts,$used" >> ci_monitor/memory_usage.csv
  sleep 5
done
