#!/usr/bin/env bash
# Runs a command as a transient systemd *user* service, so a long job survives the terminal or agent
# session that started it.  The job runs under a memory cap and under memwatch.sh, which kills it if
# the machine's available memory drops below a floor.  No sudo; nothing outside the venv is touched.
#
# Usage:   run_detached.sh <unit-name> <log-file> -- <command ...>
# Status:  systemctl --user status <unit-name>        Stop: systemctl --user stop <unit-name>
# Tunables (environment): KVT_MEM_FLOOR_GIB (default 22), KVT_MEM_CAP (default 100G),
#   KVT_THERMAL_PAUSE_C / KVT_THERMAL_RESUME_C (governor, default 70 / 60), KVT_TEMP_KILL_C (watchdog, default 92)
# The job stops if every login session of the user ends and lingering is off (loginctl show-user $USER -p Linger).
set -euo pipefail
UNIT=$1; LOG=$2
[ "${3:-}" = "--" ] || { echo "usage: run_detached.sh <unit-name> <log-file> -- <command ...>" >&2; exit 2; }
shift 3
if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "error: activate the kvtransfer venv first (source ~/.venvs/kvtransfer/bin/activate)" >&2
  exit 1
fi
HERE=$(cd "$(dirname "$0")" && pwd)
LOG=$(realpath -m "$LOG")
mkdir -p "$(dirname "$LOG")"
systemd-run --user --unit="$UNIT" --collect --quiet \
  -p MemoryMax="${KVT_MEM_CAP:-100G}" -p MemorySwapMax=0 -p WorkingDirectory="$PWD" \
  --setenv=PATH="$PATH" --setenv=VIRTUAL_ENV="$VIRTUAL_ENV" --setenv=HOME="$HOME" \
  --setenv=HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" --setenv=TOKENIZERS_PARALLELISM=false \
  --setenv=TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}" --setenv=PYTHONUNBUFFERED=1 \
  --setenv=KVT_THERMAL_PAUSE_C="${KVT_THERMAL_PAUSE_C:-70}" --setenv=KVT_THERMAL_RESUME_C="${KVT_THERMAL_RESUME_C:-60}" \
  --setenv=KVT_THERMAL_MAX_WAIT_S="${KVT_THERMAL_MAX_WAIT_S:-600}" --setenv=KVT_TEMP_KILL_C="${KVT_TEMP_KILL_C:-92}" \
  bash -c '"$0" "$1" "$2.watch" -- "${@:3}" >> "$2" 2>&1; echo "exit $?" >> "$2"' \
  "$HERE/memwatch.sh" "${KVT_MEM_FLOOR_GIB:-22}" "$LOG" "$@"
echo "started $UNIT; log $LOG; status: systemctl --user status $UNIT; stop: systemctl --user stop $UNIT"
