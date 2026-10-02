#!/usr/bin/env bash
# Runs the tiers described in docs/DGX_SPARK.md on a DGX Spark, in the foreground, under the memory
# and temperature watchdog and with the thermal governor on.  Read docs/DGX_SPARK.md section 0 first:
# run flat out, this workload powers the machine off.  For a job that must survive your terminal,
# use scripts/dgx_spark/run_detached.sh with the same kvtransfer commands instead.
# Usage: bash scripts/dgx_spark/run_tiers.sh [runs_dir]
set -euo pipefail
RUNS=${1:-runs}
HERE=$(cd "$(dirname "$0")" && pwd)
# Refuse to run outside a virtual environment: nothing here may touch the system Python or torch.
if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "error: activate the kvtransfer venv first (bash scripts/dgx_spark/setup_venv.sh; source ~/.venvs/kvtransfer/bin/activate)" >&2
  exit 1
fi
export TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}
export TOKENIZERS_PARALLELISM=false
export KVT_THERMAL_PAUSE_C=${KVT_THERMAL_PAUSE_C:-68} KVT_THERMAL_RESUME_C=${KVT_THERMAL_RESUME_C:-58}
export KVT_TEMP_KILL_C=${KVT_TEMP_KILL_C:-90}
DATA=${KVT_DATA:-$RUNS/data/fineweb-edu-head.jsonl}     # a local snapshot; streaming hangs at interpreter exit
if [ ! -f "$DATA" ]; then
  echo "error: calibration text $DATA not found. Create it with: python $HERE/snapshot_calibration_text.py $DATA" >&2
  echo "       (or set KVT_DATA to your own .jsonl/.txt file; see docs/DGX_SPARK.md section 0)" >&2
  exit 1
fi
SAFE="--batch-size 2 --bench-seq-lens 64,128,256,512,1024,2048,4096,8192 --bench-warmup 5 --bench-trials 10"
guard() { local log=$1; shift; "$HERE/memwatch.sh" "${KVT_MEM_FLOOR_GIB:-22}" "$log" -- "$@"; }
mkdir -p "$RUNS"

kvtransfer doctor

echo "== Tier 1: paper-faithful matched-KV control (Qwen3-4B -> Qwen3-8B, same tokenizer as the pod's Qwen3-4B judge)"
guard "$RUNS/tier1.watch" kvtransfer experiment --source Qwen/Qwen3-4B-Instruct-2507 --target Qwen/Qwen3-8B \
  --data "$DATA" --out "$RUNS/qwen3-4b-to-8b" --hardware dgx-spark $SAFE
BEST=$(python -c "import json; print(json.load(open('$RUNS/qwen3-4b-to-8b/report.json'))['best_k'])")
guard "$RUNS/tier1-harness.watch" kvtransfer harness --source Qwen/Qwen3-4B-Instruct-2507 --target Qwen/Qwen3-8B \
  --mapper "$RUNS/qwen3-4b-to-8b/mappers/k$BEST" --tasks arc_challenge,hellaswag,winogrande --limit 500 \
  --out "$RUNS/qwen3-4b-to-8b/harness.json"

echo "== Tier 2: the pod's escalation pair, mismatched KV (Qwen2.5-7B -> Qwen2.5-14B), research extension"
guard "$RUNS/tier2.watch" kvtransfer experiment --source Qwen/Qwen2.5-7B-Instruct --target Qwen/Qwen2.5-14B-Instruct \
  --data "$DATA" --out "$RUNS/qwen2.5-7b-to-14b" --hardware dgx-spark --allow-mismatched $SAFE
BEST=$(python -c "import json; print(json.load(open('$RUNS/qwen2.5-7b-to-14b/report.json'))['best_k'])")
guard "$RUNS/tier2-harness.watch" kvtransfer harness --source Qwen/Qwen2.5-7B-Instruct --target Qwen/Qwen2.5-14B-Instruct \
  --mapper "$RUNS/qwen2.5-7b-to-14b/mappers/k$BEST" --tasks arc_challenge,hellaswag,winogrande --limit 500 \
  --out "$RUNS/qwen2.5-7b-to-14b/harness.json"

echo "== Tier 3: escalation server for the pod's SYNTHESIZE -> ESCALATE flow (leave running; see docs/VORTEXEDGE.md)"
echo "kvtransfer serve --source Qwen/Qwen2.5-7B-Instruct --target Qwen/Qwen2.5-14B-Instruct --mapper $RUNS/qwen2.5-7b-to-14b/mappers/k$BEST --port 8811"
