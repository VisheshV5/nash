#!/usr/bin/env bash
# Train only while the laptop is plugged in.
#
#   scripts/train_when_plugged.sh <config> <run_dir> [extra train.py args...]
#
# Every 15 s: on AC power, make sure training is running (resuming from the latest checkpoint);
# on battery, stop it cleanly (SIGTERM -> checkpoint -> exit). Unplug in the morning and it
# pauses; plug in at night and it carries on. Exits when training reaches its target, or on
# SIGTERM / Ctrl+C (stopping training first). Events go to <run_dir>/watcher.log.
#
# Without pmset (Linux), the machine counts as always plugged in. For testing, point
# REGRET_FORCE_POWER_FILE at a file containing "ac" or "battery".
set -uo pipefail

if [[ $# -lt 2 ]]; then
  sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi
config="$1"; run_dir="$2"; shift 2
cd "$(dirname "$0")/.."
mkdir -p "$run_dir"
log="$run_dir/watcher.log"
child=""

say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >>"$log"; }

on_ac() {
  if [[ -n "${REGRET_FORCE_POWER_FILE:-}" ]]; then
    [[ "$(cat "$REGRET_FORCE_POWER_FILE" 2>/dev/null)" == ac ]]
    return
  fi
  command -v pmset >/dev/null 2>&1 || return 0
  pmset -g batt | grep -q "AC Power"
}

running() { [[ -n "$child" ]] && kill -0 "$child" 2>/dev/null; }

stop_training() {
  running || return 0
  local pid_file="$run_dir/train.pid"
  if [[ -f "$pid_file" ]]; then
    kill -TERM "$(cat "$pid_file")" 2>/dev/null
  else
    pkill -TERM -P "$child" 2>/dev/null  # trainer hasn't written its pid file yet
  fi
  wait "$child" 2>/dev/null
  child=""
}

trap 'say "watcher stopping"; stop_training; say "watcher stopped"; exit 0' TERM INT
echo $$ >"$run_dir/watcher.pid"
say "watcher started for $config"

while true; do
  if on_ac; then
    if [[ -n "$child" ]] && ! running; then
      wait "$child"
      rc=$?
      child=""
      if [[ $rc -eq 0 ]]; then
        say "training ended on its own (target reached, or stopped by hand); watcher exiting"
        rm -f "$run_dir/watcher.pid"
        exit 0
      elif [[ $rc -eq 2 ]]; then
        say "training refused to start (exit 2, see train.out); watcher exiting"
        rm -f "$run_dir/watcher.pid"
        exit 2
      fi
      say "training exited with code $rc; restarting in 60 s"
      sleep 60
    fi
    if ! running; then
      say "plugged in: starting training"
      caffeinate_cmd=()
      command -v caffeinate >/dev/null 2>&1 && caffeinate_cmd=(caffeinate -is)
      ${caffeinate_cmd[@]+"${caffeinate_cmd[@]}"} uv run python scripts/train.py "$config" --run-dir "$run_dir" \
        --resume "$@" >>"$run_dir/train.out" 2>&1 &
      child=$!
    fi
  elif running; then
    say "on battery: checkpointing and pausing training"
    stop_training
    say "paused"
  fi
  sleep "${REGRET_WATCH_INTERVAL:-15}" &
  wait $!
done
