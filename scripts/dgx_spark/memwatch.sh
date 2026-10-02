#!/usr/bin/env bash
# Memory and temperature watchdog for the DGX Spark.  Runs a command, samples MemAvailable and the
# hottest thermal zone every second, and kills the command (TERM, then KILL) if available memory
# drops below a floor or the machine gets too hot, so a runaway job ends instead of freezing or
# powering off a shared machine.  Reads /proc and /sys only; touches nothing else.
#
# Usage: memwatch.sh <mem_floor_gib> <log_file> -- <command ...>
# Environment: KVT_TEMP_KILL_C  kill when the hottest zone stays at or above this for 2 samples (default 92)
set -u
FLOOR_GIB=$1; LOG=$2; shift 3
FLOOR_KB=$((FLOOR_GIB * 1024 * 1024))
TEMP_KILL=${KVT_TEMP_KILL_C:-92}

hottest() {
  local max=0 t
  for f in /sys/class/thermal/thermal_zone*/temp; do
    t=$(cat "$f" 2>/dev/null) || continue
    t=$((t / 1000))
    [ "$t" -gt "$max" ] && max=$t
  done
  echo "$max"
}

stop_job() {
  echo "$(date -u +%H:%M:%S) watchdog: $1, killing $PID" >> "$LOG"
  pkill -TERM -P "$PID" 2>/dev/null; kill -TERM "$PID" 2>/dev/null
  sleep 5
  pkill -KILL -P "$PID" 2>/dev/null; kill -KILL "$PID" 2>/dev/null
  wait "$PID" 2>/dev/null
  echo "$(date -u +%H:%M:%S) watchdog: killed; minimum available was $((MIN_KB / 1024)) MiB, peak temperature $MAX_T C" >> "$LOG"
  exit 137
}

"$@" &
PID=$!
MIN_KB=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
MAX_T=$(hottest)
HOT=0
N=0
echo "$(date -u +%H:%M:%S) watchdog: pid $PID, memory floor ${FLOOR_GIB} GiB, temperature kill ${TEMP_KILL} C, start available $((MIN_KB / 1048576)) GiB at ${MAX_T} C" >> "$LOG"
while kill -0 "$PID" 2>/dev/null; do
  AVAIL_KB=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  T=$(hottest)
  [ "$AVAIL_KB" -lt "$MIN_KB" ] && MIN_KB=$AVAIL_KB
  [ "$T" -gt "$MAX_T" ] && MAX_T=$T
  if [ "$AVAIL_KB" -lt "$FLOOR_KB" ]; then
    stop_job "available $((AVAIL_KB / 1024)) MiB < floor"
  fi
  if [ "$T" -ge "$TEMP_KILL" ]; then HOT=$((HOT + 1)); else HOT=0; fi
  if [ "$HOT" -ge 2 ]; then
    stop_job "hottest zone ${T} C >= ${TEMP_KILL} C"
  fi
  N=$((N + 1))
  if [ $((N % 60)) -eq 0 ]; then
    echo "$(date -u +%H:%M:%S) watchdog: available $((AVAIL_KB / 1048576)) GiB (min $((MIN_KB / 1048576)) GiB), ${T} C (peak ${MAX_T} C)" >> "$LOG"
  fi
  sleep 1
done
wait "$PID"; RC=$?
echo "$(date -u +%H:%M:%S) watchdog: command exited $RC; minimum available was $((MIN_KB / 1048576)) GiB, peak temperature ${MAX_T} C" >> "$LOG"
exit $RC
