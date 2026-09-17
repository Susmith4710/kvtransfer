# Running kvtransfer on the NVIDIA DGX Spark

The Spark (GB10 Grace Blackwell) is one Arm CPU and one Blackwell GPU (compute capability 12.1)
sharing a single 128 GB LPDDR5x pool at ~273 GB/s. There is no separate VRAM. That changes three
things compared with the paper's 8×H100 node:

1. **Both models plus the calibration statistics must fit in ~97 GB** (80 % of the pool; boxes have
   frozen above that). `kvtransfer plan` and `kvtransfer pairs` compute this per pair.
2. **Memory reporting lies.** `nvidia-smi` shows N/A, and `torch.cuda.mem_get_info()` "free" tracks
   raw MemFree, not what is reclaimable. Use `free -h` (available) or `kvtransfer doctor`.
3. **Prefill is slower than on an H100**, so the mapper's advantage over re-prefill is at least as
   large. Decode is memory-bound at roughly 273 / (2 × params-in-billions) tokens/s in bf16.

## 1. Environment

Everything lives in a virtual environment. Nothing below touches the system Python, the system
torch, the CUDA toolkit or the driver; the venv gets its own torch (cu130 wheel) and its own copies
of every library. `run_tiers.sh` refuses to start outside a venv, and `kvtransfer doctor` warns.

```bash
git clone https://github.com/Susmith4710/kvtransfer && cd kvtransfer
bash scripts/dgx_spark/setup_venv.sh            # creates ~/.venvs/kvtransfer, installs torch cu130 + kvtransfer[data,eval,nvml]
source ~/.venvs/kvtransfer/bin/activate
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas   # env var only; only matters if something uses torch.compile

kvtransfer doctor
```

To remove it later: `rm -rf ~/.venvs/kvtransfer`. The venv's torch is a separate ~3 GB download
and does not replace or upgrade the one already on the box.

`doctor` prints the device, the memory pool, whether bf16 works, and which attention backend to use.
On the Spark that is `sdpa` (flash-attn has no sm_121 wheels; the library defaults to SDPA).

Alternative with the same isolation: NVIDIA's container, `nvcr.io/nvidia/pytorch:25.11-py3` or
newer, then `pip install -e kvtransfer[data,eval]` inside the container.

Before a big run: `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`. Consider a cgroup cap
(`systemd-run --scope -p MemoryMax=100G ...`) and disabling swap so an over-allocation fails
instead of freezing the box.

## 2. What the paper's method needs from your models

* Same tokenizer, **matched KV** (same number of KV heads and same head dimension), dense
  full attention. Depth and width may differ.
* **PyTorch checkpoints**, not GGUF. Ollama cannot export or import a KV cache, so the models
  already pulled through Ollama (`~/.ollama/models`) cannot be used directly even though they are
  the same weights. Download the Hugging Face version of each model you want to test into a
  directory of your choice (`huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir
  /data/hf/Qwen2.5-7B-Instruct`); `kvtransfer discover` shows the Ollama tag → HF id mapping.
  Sizes in bf16: Qwen2.5-7B 15 GB, Qwen2.5-14B 30 GB, Qwen3-4B 8 GB, Qwen3-8B 16 GB.

## 3. Analyse your fleet before downloading anything

```bash
kvtransfer pairs --tags "qwen2.5:14b-instruct,qwen2.5:7b-instruct,qwen3:30b-a3b-instruct-2507-q4_K_M,qwen3:4b-instruct,Llama3.1:8b,llama3.2:1b,gemma4:26b" --hardware dgx-spark
# or point it at the pod's config: kvtransfer pairs --ollama-config path/to/config.toml
```

For the inference pod's current model set the answer is:

| Pair | Category | Why |
|---|---|---|
| Qwen2.5-7B → Qwen2.5-14B | mismatched-KV | 4 vs 8 KV heads. Ridge is defined, paper never tested it |
| Qwen3-4B → Qwen3-30B-A3B | mismatched-KV | 8 vs 4 KV heads, MoE target |
| Llama-3.2-1B → Llama-3.1-8B | mismatched-KV | head dim 64 vs 128 |
| Qwen3-4B → Qwen2.5-14B | cross-family | matched geometry, shared BPE, different series |
| anything ↔ Gemma 4 | unusable | sliding-window layers |
| Qwen ↔ Llama | unusable | different tokenizers |

**There is no paper-validated pair in that set.** To test the thesis as published, add one sibling:

* `Qwen/Qwen3-8B` (36 layers, same depth as Qwen3-4B) or `Qwen/Qwen3-14B` → pairs with your Qwen3-4B
* `Qwen/Qwen2.5-32B-Instruct` → pairs with your Qwen2.5-14B (fits: 95 GB peak in two passes)
* `meta-llama/Llama-3.2-3B-Instruct` → pairs with your Llama-3.1-8B

Once checkpoints are on disk, `kvtransfer discover --models-dir /path/to/models` reads the real
configs and lists pairs; it also lists your Ollama models and their HF equivalents.

## 4. The three experiments, in order

### Tier 1: paper-faithful control (do this first)

```bash
kvtransfer experiment --source Qwen/Qwen3-4B-Instruct-2507 --target Qwen/Qwen3-8B \
    --data fineweb-edu --out runs/qwen3-4b-to-8b --hardware dgx-spark
```

Runs the whole protocol: plan → 500×1024-token calibration (stride 4) → k sweep
{1,2,4,6,8,10,12,16,20,24,all} → held-out diagnostics per k → Table 2 ablations at the best k →
reverse calibration (large→small, so the L→S direction is evaluated and multi-turn alternates)
→ latency sweep over ten lengths 64…32768 with 50 warmup + 30 timed trials and energy →
multi-turn drift → `report.md`. Every stage resumes from disk, so a crash costs only the stage in
flight. Expect one to two hours on the Spark (two calibrations); `--no-reverse` halves that, and
the plan prints an estimate.

Then downstream accuracy, the paper's actual metric:

```bash
kvtransfer harness --source Qwen/Qwen3-4B-Instruct-2507 --target Qwen/Qwen3-8B \
    --mapper runs/qwen3-4b-to-8b/mappers/k8 --tasks arc_challenge,hellaswag,winogrande --limit 500
```

prints source, target and transfer accuracy plus retention and floor-normalized retention. The
paper's Tier 1 pairs land at 73–98 % retention; below ~60 % you are looking at a Tier 2 pair.

### Tier 2: your real escalation pair (research extension)

```bash
kvtransfer experiment --source Qwen/Qwen2.5-7B-Instruct --target Qwen/Qwen2.5-14B-Instruct \
    --data fineweb-edu --out runs/qwen2.5-7b-to-14b --hardware dgx-spark --allow-mismatched
```

Same protocol; the layer-selection probe regresses each target head on all source heads because
head-to-head matching is undefined. Treat any result as new evidence, not a replication.

### Tier 3: the pod's flow end to end

```bash
kvtransfer serve --source Qwen/Qwen2.5-7B-Instruct --target Qwen/Qwen2.5-14B-Instruct \
    --mapper runs/qwen2.5-7b-to-14b/mappers/k8 --port 8765
```

See `docs/VORTEXEDGE.md` for the request flow that mirrors SYNTHESIZE → ESCALATE and how to
compare transfer against re-prefill on latency and joules.

## 5. Memory table (bf16 models, fp32 statistics, 500×1024 stride 4, batch 4)

| Pair | Models | Stats/kind | Peak | Verdict on 97 GB |
|---|---:|---:|---:|---|
| Qwen3-4B → Qwen3-8B | 23 GB | 5 GB | ~44 GB | single pass |
| Qwen3-4B → Qwen3-14B | 35 GB | 6 GB | ~58 GB | single pass |
| Qwen2.5-7B → Qwen2.5-14B | 42 GB | 3 GB | ~55 GB | single pass |
| Qwen2.5-14B → Qwen2.5-32B | 89 GB | 22 GB | ~115 GB | two passes at batch 1, tight |
| Qwen3-4B → Qwen3-30B-A3B | 64 GB | 5 GB | ~87 GB | single pass |
| Qwen3-14B → Qwen3-32B (paper's best pair) | 89 GB | 17 GB | ~112 GB | does not fit with stats on device |

`kvtransfer plan --source A --target B --hardware dgx-spark` recomputes any row. When a pair is
tight: `--batch-size 1`, `--seq-len 512`, or calibrate K and V in two passes (`calibrate(...,
kinds=("K",))` then `("V",)` from Python; the CLI's experiment runner does one pass).

## 6. Reading the numbers

* **Attention-output cosine** (per k, in `report.md`) is the paper's cross-pair predictor of
  retention (r = +0.57). R² is not (r = −0.20).
* **Logit KL / top-1 agreement** against the target's own prefill on held-out suffixes is the
  cheapest sanity check; an identity pair gives KL ≈ 0.
* **Ablation table**: "−inference RoPE" should collapse (the paper's MMLU/GSM8K → chance);
  "−all RoPE" should be within noise at 1024 tokens; k=1 should be clearly worse.
* **Latency**: the paper reports 4–25× for small→large across 64…32K tokens. On the Spark
  the absolute numbers will be higher, the ratio similar or better.
