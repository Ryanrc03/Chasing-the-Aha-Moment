#!/usr/bin/env bash
# Attach CPU-only analysis to an ALREADY RUNNING data-night queue; never restart training.
# Usage: nohup bash scripts/finish_data_night.sh QUEUE_PID > logs/data_night_finish.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results
exec 9>logs/data_night_finish.lock
flock -n 9 || { echo 'A data-night finisher is already running'; exit 1; }
queue_pid=${1:?pass the PID of bash scripts/run_data_night.sh}
[[ "$queue_pid" =~ ^[0-9]+$ ]] || { echo 'Queue PID must be numeric'; exit 1; }
python_bin=${AHA_PYTHON:-$HOME/.venvs/aha/bin/python}
max_wait=${MAX_WAIT_SECONDS:-43200}
stage() { echo "=== $(date '+%F %T %Z') $*"; }
fail() {
  stage "$1"
  "$python_bin" -c 'import json,sys;from pathlib import Path;from datetime import datetime,timezone;p=Path("results/data_night_analysis_status.json");p.write_text(json.dumps({"status":"failed","at":datetime.now(timezone.utc).isoformat(),"error":sys.argv[1]}),encoding="utf-8")' "$1"
  exit 1
}
if [ -d "/proc/$queue_pid" ]; then
  cmd=$(tr '\0' ' ' < "/proc/$queue_pid/cmdline")
  [[ "$cmd" == *"bash scripts/run_data_night.sh"* ]] || fail "PID $queue_pid is not the expected queue: $cmd"
  started=$(awk '{print $22}' "/proc/$queue_pid/stat")
  stage "Waiting for existing queue PID $queue_pid (start token $started)"
  "$python_bin" -c 'import json;from pathlib import Path;Path("results/data_night_analysis_status.json").write_text(json.dumps({"status":"waiting","queue_pid":int(__import__("sys").argv[1])}),encoding="utf-8")' "$queue_pid"
  while [ -r "/proc/$queue_pid/stat" ]; do
    current=$(awk '{print $22}' "/proc/$queue_pid/stat" 2>/dev/null) || break
    [ "$current" = "$started" ] || break
    [ "$SECONDS" -lt "$max_wait" ] || fail "Timed out waiting for queue after $max_wait seconds"
    sleep 30
  done
fi
[ -f logs/data_night.DONE ] || fail 'Queue stopped without logs/data_night.DONE; inspect logs/data_night.log'
stage 'Queue ended; validating 500 steps and 1319 predictions, then analyzing'
"$python_bin" scripts/analyze_data_night.py
stage 'Analysis complete: docs/results_2026-10-28_data_night.md'
