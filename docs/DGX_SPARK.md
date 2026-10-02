# Running kvtransfer on the NVIDIA DGX Spark

The Spark (GB10 Grace Blackwell) is one Arm CPU and one Blackwell GPU (compute capability 12.1)
sharing a single 128 GB LPDDR5x pool at ~273 GB/s. There is no separate VRAM.

> **Read section 0 first.** Run flat out, the calibration loop powers this machine off within a few
> minutes. Every long job must go through `scripts/dgx_spark/run_detached.sh`, which applies the
> thermal governor, the temperature and memory watchdog, and a memory cap.

## 0. Thermal limits: never run a sustained GPU loop ungoverned

Measured on the test machine (2026-10-01) with `scripts/dgx_spark/telemetry.py` sampling every second:

| Load | Hottest sensor | GPU power | Notes |
|---|---:|---:|---|
| Idle | 39 °C | 4 W | |
| Calibration, ungoverned, 33 s | 85 °C and rising | 65–86 W | NVML margin to the GPU limit down to 7 °C |
| Calibration, ungoverned, 1.5–4 min | machine powers off | | no kernel log, no OOM, no Xid; happened twice |
| Calibration, governed, batch 4, pause 70 / resume 60 °C | 84 °C peaks | 80–93 W bursts | stable but too close |
| Calibration, governed, **batch 2, pause 68 / resume 58 °C** | **77 °C peaks** | 80 W bursts, ~30 % duty | 31 min run, 271 pauses, stable |
| fp64 ridge solves (any k) | 66 °C | 25 W | no pauses needed |

What heats the box is bf16 forward passes and the fp32 Gram accumulation; the die gains about 20 °C
in a 3-second burst and loses about 28 °C in 5 seconds of rest, so pausing at batch boundaries is
enough. The governor (`src/kvtransfer/thermal.py`) turns itself on when the GPU is a GB10
(68 / 58 °C) and is off elsewhere; `KVT_THERMAL_PAUSE_C` overrides it and `off` disables it. It reads
`/sys/class/thermal` and sleeps, nothing more. It protects loops that reach a checkpoint (calibration
batches, solves, evaluation sequences, benchmark trials, harness requests); a single long
operation is covered only by the watchdog's temperature kill.

```bash
source ~/.venvs/kvtransfer/bin/activate
HF_HUB_OFFLINE=1 KVT_THERMAL_PAUSE_C=68 KVT_THERMAL_RESUME_C=58 KVT_TEMP_KILL_C=90 \
  scripts/dgx_spark/run_detached.sh kvt-tier1 runs/tier1.log -- \
  kvtransfer experiment --source Qwen/Qwen3-4B-Instruct-2507 --target Qwen/Qwen3-8B \
    --data runs/data/fineweb-edu-head.jsonl --out runs/qwen3-4b-to-8b --hardware dgx-spark \
    --batch-size 2 --bench-seq-lens 64,128,256,512,1024,2048,4096,8192 --bench-warmup 5 --bench-trials 10
systemctl --user is-active kvt-tier1        # the log ends with "exit <code>" when it is done
```

* `run_detached.sh` starts a transient systemd **user** service, so the job survives the terminal
  or agent session that launched it (a plain background job does not). Lingering is off on this
  box, so one login session of the user must stay open.
* `memwatch.sh` kills the job if available memory falls below 22 GiB or the hottest zone holds
  `KVT_TEMP_KILL_C` for two seconds. There is no armed hardware watchdog: a hang or thermal
  power-off needs a manual power cycle and restarts everyone's services.
* Uninterruptible single operations are the remaining risk. A 16K-token forward pass is 5 s and a
  32K one 11 s at full power; the benchmark above stops at 8,192 tokens for that reason, and uses
  5 warmup + 10 timed trials instead of the paper's 50 + 30.
* FineWeb-Edu streaming hangs or crashes at interpreter exit (leftover `datasets` threads). Use a
  local snapshot instead; `--data` accepts it. Create it once (CPU only, a few seconds, 15 MB):
  `python scripts/dgx_spark/snapshot_calibration_text.py runs/data/fineweb-edu-head.jsonl`.
  It writes the first 3,007 documents of the default config, exactly what the stream yields.

Compared with the paper's 8×H100 node, three more things change:

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

Launch it governed and detached, exactly as in section 0:

```bash
HF_HUB_OFFLINE=1 KVT_THERMAL_PAUSE_C=68 KVT_THERMAL_RESUME_C=58 KVT_TEMP_KILL_C=90 \
  scripts/dgx_spark/run_detached.sh kvt-tier1 runs/tier1.log -- \
  kvtransfer experiment --source Qwen/Qwen3-4B-Instruct-2507 --target Qwen/Qwen3-8B \
    --data runs/data/fineweb-edu-head.jsonl --out runs/qwen3-4b-to-8b --hardware dgx-spark \
    --batch-size 2 --bench-seq-lens 64,128,256,512,1024,2048,4096,8192 --bench-warmup 5 --bench-trials 10
```

Runs the whole protocol: plan → 500×1024-token calibration (stride 4) → k sweep
{1,2,4,6,8,10,12,16,20,24,all} → held-out diagnostics per k → Table 2 ablations at the best k →
reverse calibration (large→small, so the L→S direction is evaluated and multi-turn alternates)
→ latency sweep with energy → multi-turn drift → `report.md`. Every stage resumes from disk, so an
interruption costs only the stage in flight.

Measured wall-clock on the test machine, governed (2026-10-01): calibration 19 min + 3.5 min to save;
fits 9 s (k=2), 2 min (k=8), 7 min (k=12), 14.5 min (k=16), 25 min (k=20), 40 min (k=24),
24 min (k=all, one shared LU factor per kind); evaluation 1–3 min per k; ablation 15 min; reverse
45 min; benchmark 12 min. About four hours in total, of which 72 minutes were governor pauses.
`--k 1,4,8,12,16` cuts the fits to under half an hour and still brackets the optimum found here.

Then downstream accuracy, the paper's actual metric, again governed and detached:

```bash
HF_HUB_OFFLINE=1 KVT_THERMAL_PAUSE_C=68 KVT_THERMAL_RESUME_C=58 KVT_TEMP_KILL_C=90 \
  scripts/dgx_spark/run_detached.sh kvt-harness runs/harness.log -- \
  kvtransfer harness --source Qwen/Qwen3-4B-Instruct-2507 --target Qwen/Qwen3-8B \
    --mapper runs/qwen3-4b-to-8b/mappers/k12 --tasks arc_challenge,hellaswag,winogrande --limit 500 \
    --out runs/qwen3-4b-to-8b/harness.json
```

It prints source, target and transfer accuracy plus retention and floor-normalized retention, and
saves each of the three evaluations under `harness_parts/` as it finishes so a rerun resumes. The
paper's Tier 1 pairs land at 73–98 % retention; below ~60 % you are looking at a Tier 2 pair.
Results for this pair are in `docs/RESULTS_TIER1.md`.

### Tier 2: your real escalation pair (research extension)

```bash
HF_HUB_OFFLINE=1 scripts/dgx_spark/run_detached.sh kvt-tier2 runs/tier2.log -- \
  kvtransfer experiment --source Qwen/Qwen2.5-7B-Instruct --target Qwen/Qwen2.5-14B-Instruct \
    --data runs/data/fineweb-edu-head.jsonl --out runs/qwen2.5-7b-to-14b --hardware dgx-spark --allow-mismatched \
    --batch-size 2 --bench-seq-lens 64,128,256,512,1024,2048,4096,8192 --bench-warmup 5 --bench-trials 10
```

The pod runs these models from AWQ checkpoints; the method needs the bf16 Hugging Face checkpoints
(`Qwen/Qwen2.5-7B-Instruct`, 15 GB, and `Qwen/Qwen2.5-14B-Instruct`, 30 GB), which are a separate
download.

Same protocol; the layer-selection probe regresses each target head on all source heads because
head-to-head matching is undefined. Treat any result as new evidence, not a replication.

### Tier 3: the pod's flow end to end

```bash
kvtransfer serve --source Qwen/Qwen2.5-7B-Instruct --target Qwen/Qwen2.5-14B-Instruct \
    --mapper runs/qwen2.5-7b-to-14b/mappers/k8 --port 8811
```

Pick a port nobody else on the box uses (8765 was already in use on the test machine). The server is not thermally
governed between requests: drive it with spaced requests, not a tight loop.

See `docs/VORTEXEDGE.md` for the request flow that mirrors SYNTHESIZE → ESCALATE and how to
compare transfer against re-prefill on latency and joules.

## 5. Memory table (bf16 models, fp32 statistics, 500×1024 stride 4, batch 2, GiB)

The experiment collects three statistic kinds (K, V and the rotated keys for the RoPE ablation), not
two, and the largest ridge solve needs its own headroom. `kvtransfer plan` now reports both.

| Pair | Models | Stats/kind | Calibration peak (3 kinds) | Fit peak at k=all | Verdict on 97 GiB |
|---|---:|---:|---:|---:|---|
| Qwen3-4B → Qwen3-8B | 22.7 | 10.1 | 56 | 61 | fits (measured, see below) |
| Qwen3-4B → Qwen3-14B | 35.0 | 10.7 | 70 | 62 | fits |
| Qwen2.5-7B → Qwen2.5-14B | 41.7 | 3.4 | 55 | 15 | fits |
| Qwen3-4B → Qwen3-30B-A3B | 64.3 | 8.4 | 93 | 56 | tight; use `--no-ablation` (84) |
| Qwen2.5-14B → Qwen2.5-32B | 88.5 | 21.0 | 113 | 117 | does not fit |
| Qwen3-14B → Qwen3-32B (paper's best pair) | 88.5 | 16.2 | 108 | 86 | does not fit |

Measured for Qwen3-4B → Qwen3-8B on the test machine: calibration left 46 GiB available system-wide; the
k=all fit left only 26 GiB (PyTorch kept freed blocks cached next to the statistics). The shared
solve has since been slimmed to one gather, an in-place centring and one LU factor; re-measure
before relying on the planner's figure for that step. The runner unloads the models during solves
and loads one mapper at a time, so the peak is the largest single stage, not the sum.

Truly free memory sits near 1 GiB during a run because the page cache fills the pool; available
memory (`free -h`, or the watchdog log) is the number to read. The runner releases the page cache
of its own model and statistics files with `posix_fadvise`, which needs no privileges.

`kvtransfer plan --source A --target B --hardware dgx-spark` recomputes any row. When a pair is
tight: `--no-ablation` (two kinds), `--seq-len 512`, or a smaller k sweep with `--k`.

## 6. Reading the numbers

* **Attention-output cosine** (per k, in `report.md`) is the paper's cross-pair predictor of
  retention (r = +0.57). R² is not (r = −0.20).
* **Logit KL / top-1 agreement** against the target's own prefill on held-out suffixes is the
  cheapest sanity check; an identity pair gives KL ≈ 0.
* **Ablation table**: "−inference RoPE" should collapse (the paper's MMLU/GSM8K → chance);
  "−all RoPE" should be within noise at 1024 tokens; k=1 should be clearly worse.
* **Latency**: the paper reports 4–25× for small→large across 64…32K tokens. On the Spark
  the absolute numbers will be higher, the ratio similar or better.
