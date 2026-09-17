# kvtransfer

Cross-model KV cache transfer within an LLM family, on Hugging Face `transformers` models.

An implementation of **Heo et al., "Cross-Model KV Cache Transfer in LLM Families: A Closed-Form
Linear Mapping for Prefill Reuse" (NVIDIA, arXiv:2608.03893, Aug 2026)**. A small model prefills a
prompt; a closed-form per-head ridge mapper converts its KV cache into the format a larger (or
smaller) sibling expects; the sibling decodes without re-prefilling. The paper reports 73–98 %
accuracy retention on four of six matched-KV pairs and mapper application 2.7–25× faster than
re-prefill (Qwen3 14B→32B at 32K tokens: 278 ms vs 6,975 ms).

The paper released no code and integrates with no serving engine: its experiments are plain
PyTorch + `transformers` (bf16, `flash_attention_2` for the re-prefill baseline, eager mode for
the mapper) on an 8×H100 node, scored with lm-evaluation-harness. This library follows the same
stack, which is the only one that works with plain downloaded checkpoints today.

## What the method requires

* **Same family, matched KV**: source and target share KV head count and per-head dimension
  (Qwen3 0.6B/1.7B/4B/8B/14B/32B, Llama 3.1 8B/70B, Ministral 3 3B/8B/14B all have 8 KV heads × 128).
  Depth and width may differ. Mismatched pairs are untested by the paper; `--allow-mismatched`
  lets you try anyway.
* **Shared tokenizer** (checked).
* **Dense full attention** with a model-level rotary embedding (Llama / Qwen2 / Qwen3 / Mistral-style).
  Sliding-window or hybrid SSM models are out of scope, as in the paper.
* A mapper is **directional** and specific to one checkpoint pair. A router over P models needs up
  to P(P−1) mappers of 1–3 B parameters (4–12 GB fp32, half in bf16) each.

## Install

```bash
pip install -e kvtransfer            # torch, numpy, safetensors, transformers
pip install -e "kvtransfer[data]"    # + datasets, for FineWeb-Edu streaming
pytest kvtransfer                    # offline CPU suite, tiny random models, ~3 s
```

## Testing on a DGX Spark against a specific fleet

* `docs/DGX_SPARK.md`: environment, memory rules for unified memory, which pairs fit, the three
  experiments to run in order, how to read the numbers.
* `docs/VORTEXEDGE.md`: where transfer fits a memory-first pod's SYNTHESIZE -> ESCALATE flow, why
  Ollama/GGUF cannot be used directly, prefix sharing, the escalation server, what integration would look like.
* `docs/PAPER_VERIFICATION.md`: every paper claim mapped to the code and to the test that checks it.
* `scripts/dgx_spark/run_tiers.sh`: the three tiers end to end.

```bash
kvtransfer doctor                                       # GPU, unified memory, torch/cuda, attention backend
kvtransfer pairs --tags "qwen2.5:7b-instruct,qwen2.5:14b-instruct,qwen3:4b-instruct" --hardware dgx-spark
kvtransfer discover --models-dir /data/models            # real configs + Ollama store -> pairs
kvtransfer plan --source Qwen/Qwen3-4B-Instruct-2507 --target Qwen/Qwen3-8B --hardware dgx-spark
kvtransfer experiment --source Qwen/Qwen3-4B-Instruct-2507 --target Qwen/Qwen3-8B --out runs/q3-4b-8b
kvtransfer harness    --source ... --target ... --mapper runs/q3-4b-8b/mappers/k8 --tasks arc_challenge,hellaswag
kvtransfer serve      --source ... --target ... --mapper runs/q3-4b-8b/mappers/k8 --port 8765
```

`experiment` runs the paper's whole protocol on one pair (plan, calibrate, k sweep, held-out
diagnostics, Table 2 ablations, latency with energy, multi-turn drift) and writes `report.md`;
`harness` gives downstream accuracy retention through lm-evaluation-harness, the paper's metric;
`serve` is a small-model-then-escalate service that reports skipped tokens, mapper time and joules
per request. Mismatched-KV pairs (e.g. Qwen2.5-7B -> 14B, 4 vs 8 KV heads) run with
`--allow-mismatched` as a research extension the paper did not test.

## Command line

```bash
# 0. Is the pair eligible, and how big will the mapper be?
kvtransfer check --source Qwen/Qwen3-1.7B --target Qwen/Qwen3-4B --k 1,4,8

# 1. Calibrate once (paper recipe: 500 x 1,024 FineWeb-Edu tokens, stride 4) and fit any k.
#    --stats keeps the cross-layer moments so later fits for other k need no models.
kvtransfer fit --source Qwen/Qwen3-1.7B --target Qwen/Qwen3-4B \
    --data fineweb-edu --n-seqs 500 --seq-len 1024 --stride 4 \
    --k 1,4,8,all --lam 0.01 --out mappers/qwen3-1.7b-to-4b --stats stats/qwen3-1.7b-to-4b

# 2. Diagnostics on held-out text: held-out R2, attention-output cosine, logit KL, top-1 agreement.
kvtransfer eval --source Qwen/Qwen3-1.7B --target Qwen/Qwen3-4B \
    --mapper mappers/qwen3-1.7b-to-4b/k8 --data fineweb-edu --n-seqs 32 --seq-len 1024 --suffix-len 32

# 3. Latency: mapper application vs. target re-prefill (transformer body, no LM head).
kvtransfer bench --source Qwen/Qwen3-1.7B --target Qwen/Qwen3-4B \
    --mapper mappers/qwen3-1.7b-to-4b/k8 --seq-lens 64,512,2048,8192

# 4. Try it: source prefills, target decodes from the mapped cache (and, with --compare, both standalone).
kvtransfer generate --source Qwen/Qwen3-1.7B --target Qwen/Qwen3-4B \
    --mapper mappers/qwen3-1.7b-to-4b/k8 --prompt "Explain why the sky is blue." --compare
```

Local checkpoints work anywhere a model id does. `--data` also accepts `hf:<dataset>[:config][:split]`
or a local `.txt` (blank-line separated documents) / `.jsonl` (`text` field) file.

## Python API

```python
import torch
from kvtransfer import (calibrate, Mapper, CrossModelTransfer, Session, evaluate, benchmark,
                        load_model, load_tokenizer)
from kvtransfer.data import batches, iter_dataset, token_sequences

src, tgt = load_model("Qwen/Qwen3-1.7B"), load_model("Qwen/Qwen3-4B")   # bf16 on CUDA, fp32 on CPU
tok = load_tokenizer("Qwen/Qwen3-1.7B")

seqs = token_sequences(iter_dataset("fineweb-edu"), tok, seq_len=1024, n_seqs=500)
stats = calibrate(src, tgt, batches(seqs, 4), stride=4)     # one streaming pass over both models
stats.save("stats/qwen3-1.7b-to-4b")                         # reusable for any k

mapper = Mapper.fit(stats, k=8, lam=0.01)                    # closed-form, seconds
mapper.save("mappers/qwen3-1.7b-to-4b/k8")                   # safetensors + json
print(mapper.summary())

xfer = CrossModelTransfer(src, tgt, mapper)
ids = tok("Explain why the sky is blue.", return_tensors="pt")["input_ids"]
res = xfer.generate(ids, max_new_tokens=64, hold_back=1)     # source prefill -> map -> target decode
print(tok.decode(res.tokens[0, res.n_prompt:]))

# lower-level: map an existing source cache into a target cache
_, src_cache = xfer.source_prefill(ids)
tgt_cache = xfer.map_cache(src_cache, n_tokens=ids.shape[1] - 1)

# multi-turn switching (paper Sec. 4.6) with a mapper for each direction
sess = Session({"small": src, "large": tgt},
               {("small", "large"): mapper, ("large", "small"): Mapper.load("mappers/qwen3-4b-to-1.7b/k8")},
               start="small")
sess.feed(ids); sess.generate(32); sess.switch_to("large"); sess.feed(more_ids); sess.generate(32)
```

## What is implemented, mapped to the paper

| Paper | Here |
|---|---|
| Sec. 2.3 single-source OLS probe, head-averaged R² heatmap | `select.probe_r2`, `selection_score` |
| Sec. 3.1 per-head centered ridge, λ = 0.01, closed form (Eq. 4) | `ridge.MomentAccumulator.solve`, `Mapper.fit` |
| Sec. 3.2 top-k source layers per target layer, shared across heads, selected on mean of K_stripped and V R² | `select.top_k_layers`, `Mapper.fit(k=...)` |
| Sec. 3.3 strip source RoPE, map, re-apply target RoPE; values mapped directly | `rope.RopeCodec` (cos/sin taken from each model's own rotary module, so YaRN/llama3 scaling is exact) |
| Calibration: 500 × 1,024 FineWeb-Edu tokens, stride 4 | `calibration.calibrate`, `data.iter_fineweb_edu` |
| Appendix D mapper size formula | `Mapper.formula_params`, `kvtransfer check` |
| Sec. 4.5 attention-output cosine as the retention predictor; R² as a within-pair diagnostic | `metrics.evaluate` (also logit KL and top-1 agreement vs. the target's own prefill) |
| Sec. 4.6 multi-turn handoff | `transfer.Session` |
| Sec. 4.7 mapper vs. re-prefill latency | `bench.benchmark` |
| Sec. 4.3 / Table 2 ablations (-all RoPE, -inference RoPE, k=1) | `Mapper.fit(key_space="rope")`, `Mapper.ablate_inference_rope()`, `experiment` stage `ablation` |
| Sec. 4.1 benchmarks and retention / floor-normalized retention | `lm_eval_adapter.TransferLM` (`--model kvtransfer` in lm-eval), `retention_table`, `kvtransfer harness` |
| Sec. 4.2 / 4.5 large-to-small direction | `experiment` stage `reverse` (calibrates target->source, evaluates, enables alternating multi-turn) |
| Sec. 4.4 MLP mapper for the pairs where ridge fails | not implemented |
| Sec. 4.5 K/V error-concentration diagnostics, App. B greedy forward selection, App. C lambda/N/domain sweeps, App. D prefix-conditioned WikiText perplexity, App. H leave-one-out re-selection | not implemented (see docs/PAPER_VERIFICATION.md) |

One implementation choice the paper leaves implicit: to get the target's first logit you need one
target forward pass, so the last `hold_back` prompt tokens (default 1) are not mapped but run through
the target on top of the mapped prefix.

### How calibration is stored

`calibrate` never stores tokens. For each cache kind it accumulates shifted second moments between
*every* source layer and *every* target layer (all heads concatenated), so layer selection and the
ridge for **any** k are sub-block solves of the same statistics. Sweeping k as the paper does is free
after one pass. The cost is memory, per kind and in fp32:

| Pair | Gram (L_s·n_kv·d_h)² | Cross (L_s·n_kv·d_h)·(L_t·n_kv·d_h) |
|---|---:|---:|
| Qwen3 0.6B→1.7B (28→28) | 3.3 GB | 3.3 GB |
| Qwen3 1.7B→4B (28→36) | 3.3 GB | 4.2 GB |
| Qwen3 8B→14B (36→40) | 5.4 GB | 6.0 GB |
| Qwen3 14B→32B (40→64) | 6.7 GB | 10.7 GB |

`kvtransfer check` prints these. Put them on a GPU with `--stats-device cuda`, or run K and V in
separate passes (`calibrate(..., kinds=("K",))`) when host memory is tight. The paper's own numbers
came from an 8×H100 node at roughly one hour per pair; a 0.6B→1.7B or 1.7B→4B pair fits on a single
24 GB consumer GPU (both models in bf16 plus statistics).

## Using this with an inference engine

The mapper is a pure tensor operation: `Mapper.apply_kv` takes a list over source layers of
`(K_rope, V)` tensors shaped `[B, n_kv, T, d_h]` and returns the same for the target. Anything that
can export and import per-layer KV tensors can use it. Today the only supported runtime is the HF
`DynamicCache` path in `transfer.py`, which is what the paper used. For vLLM or SGLang you would
implement a KV connector that writes the mapped tensors into the paged pool; for llama.cpp you would
first have to check whether the linear structure survives quantized KV, which nobody has measured.

## Read the diagnostics honestly

The paper's own finding (Sec. 4.5) is that calibration R² does **not** predict downstream retention
across pairs; attention-output cosine does better (r = +0.57). Two Ministral 3 pairs fit with
respectable R² and still lost more than half their accuracy. Treat `kvtransfer eval` as a screening
tool and confirm on your task before routing traffic through a mapper.

## License

Apache-2.0. Algorithm credit: Heo et al., arXiv:2608.03893.
