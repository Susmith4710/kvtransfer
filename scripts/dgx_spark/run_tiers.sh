#!/usr/bin/env bash
# Runs the three tiers described in docs/DGX_SPARK.md on a DGX Spark.  Edit MODELS_DIR / pairs as needed.
# Usage: bash scripts/dgx_spark/run_tiers.sh [runs_dir]
set -euo pipefail
RUNS=${1:-runs}
export TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}
export TOKENIZERS_PARALLELISM=false

kvtransfer doctor

echo "== Tier 1: paper-faithful matched-KV control (Qwen3-4B -> Qwen3-8B, same tokenizer as the pod's Qwen3-4B judge)"
kvtransfer experiment --source Qwen/Qwen3-4B-Instruct-2507 --target Qwen/Qwen3-8B \
  --data fineweb-edu --out "$RUNS/qwen3-4b-to-8b" --hardware dgx-spark
kvtransfer harness --source Qwen/Qwen3-4B-Instruct-2507 --target Qwen/Qwen3-8B \
  --mapper "$RUNS/qwen3-4b-to-8b/mappers/k8" --tasks arc_challenge,hellaswag,winogrande --limit 500 \
  --out "$RUNS/qwen3-4b-to-8b/harness.json"

echo "== Tier 2: the pod's escalation pair, mismatched KV (Qwen2.5-7B -> Qwen2.5-14B), research extension"
kvtransfer experiment --source Qwen/Qwen2.5-7B-Instruct --target Qwen/Qwen2.5-14B-Instruct \
  --data fineweb-edu --out "$RUNS/qwen2.5-7b-to-14b" --hardware dgx-spark --allow-mismatched
kvtransfer harness --source Qwen/Qwen2.5-7B-Instruct --target Qwen/Qwen2.5-14B-Instruct \
  --mapper "$RUNS/qwen2.5-7b-to-14b/mappers/k8" --tasks arc_challenge,hellaswag,winogrande --limit 500 \
  --out "$RUNS/qwen2.5-7b-to-14b/harness.json"

echo "== Tier 3: escalation server for the pod's SYNTHESIZE -> ESCALATE flow (leave running; see docs/VORTEXEDGE.md)"
echo "kvtransfer serve --source Qwen/Qwen2.5-7B-Instruct --target Qwen/Qwen2.5-14B-Instruct --mapper $RUNS/qwen2.5-7b-to-14b/mappers/k8 --port 8765"
