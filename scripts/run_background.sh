#!/usr/bin/env bash
# Launch training in the background so it survives closing the terminal or an SSH session.
#
#   scripts/run_background.sh configs/leduc.yaml [run_dir] [extra train.py args...]
#
# Uses tmux when available (attach later with `tmux attach -t <session>`), otherwise nohup.
# On macOS, wraps the run in `caffeinate -is` so the laptop doesn't sleep (keep it plugged in).
# Always passes --resume, so re-running this after a crash or reboot picks up where it left off.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi

config="$1"; shift
name="$(basename "${config%.*}")"
run_dir="${1:-runs/$name}"; [[ $# -gt 0 ]] && shift
cd "$(dirname "$0")/.."
mkdir -p "$run_dir"

cmd=(uv run python scripts/train.py "$config" --run-dir "$run_dir" --resume "$@")
if command -v caffeinate >/dev/null 2>&1; then
  cmd=(caffeinate -is "${cmd[@]}")
fi
out="$run_dir/train.out"
session="regret-$(basename "$run_dir")"

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session '$session' already exists: tmux attach -t $session" >&2
    exit 1
  fi
  printf -v quoted '%q ' "${cmd[@]}"
  tmux new-session -d -s "$session" "$quoted 2>&1 | tee -a $(printf %q "$out")"
  echo "started in tmux session '$session'"
  echo "  attach:  tmux attach -t $session     (detach again with Ctrl+B then D)"
else
  nohup "${cmd[@]}" >>"$out" 2>&1 &
  echo "started with nohup (pid $!); output in $out"
fi
echo "  status:  uv run python scripts/status.py $run_dir"
echo "  stop:    kill -TERM \$(cat $run_dir/train.pid)   (checkpoints, then exits)"
