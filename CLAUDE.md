# kvtransfer: working notes for Claude Code sessions

Read this first. It carries the context from the session that built this repo.

## What this is

An implementation of Heo et al., "Cross-Model KV Cache Transfer in LLM Families: A Closed-Form
Linear Mapping for Prefill Reuse" (NVIDIA, arXiv:2608.03893). A small model prefills a prompt, a
closed-form ridge mapper converts its KV cache into a larger sibling's cache, the sibling decodes
without re-prefilling. The paper released no code and integrates with no serving engine; its
experiments are plain PyTorch + Hugging Face transformers, and so is this library.

Docs: `README.md` (API + CLI), `docs/DGX_SPARK.md` (runbook for this machine), `docs/VORTEXEDGE.md`
(how it fits the inference pod), `docs/PAPER_VERIFICATION.md` (every paper claim -> code -> test).

## The machine and the hard rules

* Target hardware: **NVIDIA DGX Spark** (GB10, Arm + Blackwell sm_121, 128 GB unified memory,
  CUDA 13 only). Plan against 80 % of the pool; `nvidia-smi` shows memory as N/A; use `sdpa`, not
  flash-attn; torch must be a cu130 wheel.
* **Everything runs inside a virtual environment.** Never install, upgrade or remove anything in
  the system Python, system torch, CUDA toolkit or driver. Use `scripts/dgx_spark/setup_venv.sh`
  (creates `~/.venvs/kvtransfer`) and `source ~/.venvs/kvtransfer/bin/activate`. `run_tiers.sh`
  refuses to run outside a venv; `kvtransfer doctor` warns.
* **Sustained GPU load powers this machine off.** Unthrottled calibration took the SoC from 39 C to
  85 C in 33 s and shut the box down twice (2026-09-30), with no log entry and a manual power cycle
  each time. The thermal governor (`src/kvtransfer/thermal.py`) is on by default on the GB10; never
  disable it, never write a new sustained GPU loop without a `thermal.checkpoint()`, and use
  `--batch-size 2`. Details and measurements: `docs/DGX_SPARK.md` section 0.
* **Launch every long job with `scripts/dgx_spark/run_detached.sh <unit> <log> -- <command>`.** It
  runs the job as a systemd user service (a background job of an agent session dies with the
  session), under the governor, a 100 GiB cap, and `memwatch.sh`, which kills the job on low memory
  or high temperature. Check with `systemctl --user is-active <unit>`; the log ends with `exit <code>`.
* Do not modify the VortexEdge inference-pod repository. This repo is a standalone test bed.
* The box is shared. Do not touch, query or restart anything other users rely on: the inference-pod
  project and its monitoring containers (Prometheus, Grafana), the DCGM exporter, Ollama, other
  users' processes. For telemetry use `scripts/dgx_spark/telemetry.py`, which reads only this machine's
  own sensors.
* The pod's production engine is Ollama (GGUF). Ollama cannot export or import a KV cache, so the
  models pulled through Ollama cannot be used here; the Hugging Face checkpoints must be downloaded
  separately (`huggingface-cli download <id> --local-dir <dir>`).

## The user's models and what they mean for the paper

Pod config (Ollama tags): qwen2.5:14b-instruct (llm/slm/reasoning), qwen2.5:7b-instruct
(ragqa/atomicfacts), qwen3:30b-a3b-instruct-2507-q4_K_M (synthesis/passagener), qwen3:4b-instruct,
Llama3.1:8b (post-process judges), llama3.2:1b (queryner), gemma4:26b (judge).

`kvtransfer pairs --tags "..." --hardware dgx-spark` says: **no paper-validated matched-KV pair in
that set.** Qwen2.5-7B -> 14B is 4 vs 8 KV heads (mismatched, research extension via
`--allow-mismatched`); Qwen3-4B -> Qwen3-30B-A3B is 8 vs 4; Llama 3.2-1B -> 3.1-8B differs in head
dim; Gemma 4 has sliding-window layers (out of scope); Qwen <-> Llama tokenizers differ. A
paper-faithful control needs one sibling download: `Qwen/Qwen3-8B` (pairs with Qwen3-4B) or
`Qwen/Qwen2.5-32B-Instruct` (pairs with Qwen2.5-14B, tight on memory).

## The plan (docs/DGX_SPARK.md section 4)

1. Tier 1, paper-faithful control: `kvtransfer experiment --source Qwen/Qwen3-4B-Instruct-2507
   --target Qwen/Qwen3-8B --out runs/qwen3-4b-to-8b --hardware dgx-spark`, then `kvtransfer harness`.
   Done; see `docs/RESULTS_TIER1.md`.
2. Tier 2, the pod's real escalation pair: same with Qwen2.5-7B -> Qwen2.5-14B and `--allow-mismatched`.
3. Tier 3, the pod's flow: `kvtransfer serve` (small model answers, large model continues from the
   mapped cache; reports skipped tokens, mapper ms, joules).

Local checkpoint paths can be passed anywhere an HF id is accepted. `kvtransfer discover
--models-dir <dir>` reads real configs and lists pairs.

## State of the code

* 99 offline tests (`pytest`, CPU, tiny random models, ~10 s). They verify the mathematics and the
  plumbing (identity pair reproduces native logits exactly, also beyond the calibration length),
  not the paper's accuracy numbers.
* **Tier 1 has run on the real pair** (Qwen3-4B -> Qwen3-8B, 2026-10-01). Results and how they
  compare with the paper are in `docs/RESULTS_TIER1.md`; raw outputs are under
  `runs/qwen3-4b-to-8b/` (not committed). Best k is 12 by held-out attention-output cosine (0.899).
* Tier 2 (Qwen2.5-7B -> 14B, mismatched KV) and Tier 3 (`kvtransfer serve`) have not run. Tier 2
  needs the bf16 Hugging Face checkpoints, which are not on disk (the pod uses AWQ builds).
* The experiment runner is memory-disciplined for unified memory: models are unloaded during ridge
  solves, mappers are loaded one at a time, statistics are dropped when no longer needed, and a fit
  whose target layers share their source layers (always true for k = all) factors the Gram once.
* An adversarial review against the paper text was done; its findings are fixed and listed in
  `docs/PAPER_VERIFICATION.md`, along with what is deliberately not implemented (MLP mapper,
  greedy selection, lambda/N/domain sweeps, error-concentration diagnostics, WikiText prefix
  perplexity protocol).

## Conventions

* Keep changes minimal and tested; run `pytest` before committing.
* Never commit `runs/`, `mappers/`, `stats/` (gitignored) or model weights.
* Commit messages: plain description of the change; no model identifiers in code or commits.
