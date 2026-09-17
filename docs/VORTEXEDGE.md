# Using kvtransfer with a memory-first inference pod (VortexEdge)

Nothing in the pod has to change to run the experiments. This document explains where the
paper's idea fits the pod's request flow, what the pod's current engine cannot do, and how to
measure the benefit with the pieces in this repo.

## Where transfer fits

The pod decides per query between `CACHE_HIT`, `SYNTHESIZE` (a small model writes a grounded
answer from a memory briefing) and `ESCALATE` (a larger model reasons from scratch). The escalation
path is exactly the paper's cascade scenario: the small model has already prefilled the system
prompt plus briefing, and the larger model then pays that prefill again. Cross-model KV transfer
lets the larger model start from the small model's cache.

What it saves is the large model's **prefill** of the shared prefix. What it costs is one batched
matmul per target layer, plus holding a 1–3 B parameter mapper per (small, large) direction. Decode
speed is unchanged.

## What the current engine cannot do

The pod serves Ollama (llama.cpp, GGUF). Ollama has no API to read a model's KV cache out or to
write one in, and its caches are quantized. The paper's method, and this library, need a PyTorch
model whose cache tensors are addressable. So the test setup runs the (small, large) pair in
Hugging Face transformers on the Spark, next to Ollama, not inside it. The pod's vLLM alternative
would need a custom KV connector; that is a separate project and not part of this repo.

## Prefix sharing is the whole game

Transfer only skips tokens the two models have **in common at the start of the prompt**. In the
pod, the system prompt and the briefing are typically identical between synthesis and escalation,
while the instruction tail differs. The server in this repo maps the longest common token prefix
and lets the large model prefill only the tail. Two practical rules:

* Put everything shared first: system prompt, memory briefing, retrieved passages, conversation so
  far. Put the role-specific instruction last.
* Both roles must use the same tokenizer family and the same chat template up to the divergence
  point. `kvtransfer check` verifies tokenizer compatibility.

## Which of the pod's models can pair

See `docs/DGX_SPARK.md` §3: with the current config (Qwen2.5 7B/14B, Qwen3 4B/30B-A3B, Llama 3.1
8B, Llama 3.2 1B, Gemma 4 26B) there is no paper-validated matched-KV pair. Qwen2.5-7B → 14B is the
natural escalation pair and is a mismatched-KV research extension. Adding Qwen3-8B or Qwen3-14B
gives a paper-faithful pair with the Qwen3-4B judge.

## Measuring the benefit the way the pod measures things

The pod reports joules, average GPU watts and tokens/s per answer via NVML. `kvtransfer serve`
reports the same per request (`pynvml` installed), plus prefill/mapper/decode milliseconds and how
many tokens were skipped.

```bash
kvtransfer serve --source Qwen/Qwen2.5-7B-Instruct --target Qwen/Qwen2.5-14B-Instruct \
    --mapper runs/qwen2.5-7b-to-14b/mappers/k8 --port 8765
```

Then, from any client (the pod's benchmark harness, curl, Python):

```bash
# 1. small model answers (SYNTHESIZE); its cache is kept for the session
curl -s localhost:8765/v1/generate -d '{"session":"q42","role":"source","chat":true,
  "system":"<system prompt + memory briefing>","prompt":"<question>","max_new_tokens":128}'

# 2. escalate with transfer: large model continues from the mapped cache
curl -s localhost:8765/v1/generate -d '{"session":"q42","role":"escalate","chat":true,
  "system":"<same system prompt + briefing>","prompt":"<question + reasoning instruction>","max_new_tokens":256}'

# 3. the baseline the pod does today: large model re-prefills everything
curl -s localhost:8765/v1/generate -d '{"session":"q42","role":"escalate","baseline":true,"chat":true,
  "system":"<same>","prompt":"<same>","max_new_tokens":256}'
```

Compare `timing.prefill_ms`, `timing.skipped_tokens`, `timing.mapper_ms`, `energy.joules` between
2 and 3, and compare the two answers for quality (the pod's judge model can score them).

## If the results are good: what integration would look like

The pod selects an engine through a factory (`ollama` or `vllm`). A third engine that talks to
`kvtransfer serve` (or embeds `kvtransfer.serve.Escalator` in-process) would implement the same
`generate` interface and add one call: on escalation, pass the session id so the large model
continues from the small model's cache. The mapper artifact (4–12 GB) can live on disk and be
memory-mapped; on the Spark's unified memory that is the same pool either way.

Keep the pod's fallback: if the session has no source cache, or the common prefix is empty, the
server already re-prefills, and reports that in `timing.mode`.

## Honest limits

* Quality retention is pair-specific. The paper's own two failure pairs looked fine on R² and lost
  half their accuracy. Gate on `kvtransfer harness` retention and the attention-output cosine
  before routing real traffic.
* A mapper is tied to one checkpoint pair and direction. Changing either model means recalibrating
  (about an hour on the Spark).
* Large-to-small transfer (escalate, then hand back to the small model for cheap follow-ups) is
  supported (`Session` in `transfer.py`), and the paper reports linearly growing drift in that
  direction; measure it with the multi-turn stage before relying on it.
