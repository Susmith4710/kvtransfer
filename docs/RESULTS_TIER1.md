# Tier 1 results: Qwen3-4B → Qwen3-8B on the DGX Spark

First run of the paper's protocol on real checkpoints (2026-10-01, one DGX Spark). Source
`Qwen/Qwen3-4B-Instruct-2507`, target `Qwen/Qwen3-8B`, both 36 layers with 8 KV heads × 128, bf16.
Calibration: 500 × 1,024 FineWeb-Edu tokens, stride 4 (128,000 tokens per head), λ = 0.01.
Raw outputs are in `runs/qwen3-4b-to-8b/` (not committed): `report.md`, `eval_k*.json`,
`ablation.json`, `bench.json`, `multiturn.json`, `harness.json`.

## Headline

| Question | Answer |
|---|---|
| Does the target decode usefully from the mapped cache? | Yes. 95.4 % average accuracy retention (91.5 % floor-normalized) at k = 12 |
| Best k | 12 by held-out attention-output cosine (0.899); quality falls for k ≥ 20 |
| Is the mapper faster than re-prefill? | Yes, but only about 2× on this pair, at every length tested |
| Large → small direction | Works: cosine 0.878, top-1 87.4 % |
| Multi-turn drift | None measurable over 10 alternating handoffs (KL slope −0.002 per turn) |

## Downstream accuracy (the paper's metric)

lm-evaluation-harness, first 500 documents per task, log-likelihood scoring, k = 12 mapper,
`hold_back` = 1. Standard error is about ±2.2 points per cell.

| Task | Metric | Source alone | Target alone | Transfer | Retention | Floor-normalized |
|---|---|---:|---:|---:|---:|---:|
| ARC-Challenge | acc_norm | 0.562 | 0.558 | 0.540 | 96.8 % | 94.2 % |
| HellaSwag | acc_norm | 0.580 | 0.642 | 0.582 | 90.7 % | 84.7 % |
| WinoGrande | acc | 0.700 | 0.684 | 0.676 | 98.8 % | 95.7 % |
| **Average** | | | | | **95.4 %** | **91.5 %** |

The paper's working pairs retain 73–98 % (its best, Qwen3 14B → 32B, 97.6 %). This pair sits near
the top of that range.

**Read this with one caveat.** On ARC-Challenge and WinoGrande the 4B source scores the same as the
8B target within noise, so high retention there says the transfer does no harm, not that it carries
over an advantage. HellaSwag is the one task where the target is clearly ahead (0.642 against
0.580), and there the transfer result (0.582) lands at the source's level. This pair is a control
for the method, not evidence that escalation through a mapped cache keeps the larger model's edge.
That needs a pair with a real accuracy gap.

## Held-out diagnostics across k

32 held-out sequences, 992-token mapped prefix, 32-token suffix decoded by the target.
Without any transfer, the source's own next-token choice matches the target's 75.0 % of the time.

| k | Mapper size (fp32) | In-sample R² K / V | Attention cosine | Worst layer | KL | Top-1 agreement |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.3 GiB | 0.778 / 0.642 | 0.824 | 0.569 | 0.173 | 83.8 % |
| 2 | 0.6 GiB | 0.800 / 0.674 | 0.857 | 0.663 | 0.139 | 84.1 % |
| 4 | 1.1 GiB | 0.820 / 0.706 | 0.881 | 0.722 | 0.115 | 85.9 % |
| 6 | 1.7 GiB | 0.833 / 0.726 | 0.887 | 0.735 | 0.103 | 87.1 % |
| 8 | 2.3 GiB | 0.842 / 0.741 | 0.890 | 0.749 | 0.096 | 87.0 % |
| 10 | 2.8 GiB | 0.848 / 0.751 | 0.889 | 0.749 | 0.091 | 87.0 % |
| **12** | **3.4 GiB** | 0.854 / 0.761 | **0.899** | **0.750** | **0.090** | 87.1 % |
| 16 | 4.5 GiB | 0.864 / 0.776 | 0.896 | 0.606 | 0.090 | 87.6 % |
| 20 | 5.6 GiB | 0.872 / 0.790 | 0.879 | 0.507 | 0.093 | 87.3 % |
| 24 | 6.8 GiB | 0.879 / 0.802 | 0.863 | 0.505 | 0.097 | 87.2 % |
| 36 (all) | 10.1 GiB | 0.899 / 0.834 | 0.744 | 0.461 | 0.176 | 84.3 % |

In-sample R² rises with every added layer while held-out quality peaks at k = 12 and then falls:
large k overfits. The paper found the same shape for its Qwen3 pairs (best k of 8 and 12) and makes
the same point that R² does not predict transfer quality. k = 4 already reaches 0.881 with a mapper
a third the size of k = 12.

## Ablation at k = 12 (paper Table 2)

| Variant | Attention cosine | KL | Top-1 |
|---|---:|---:|---:|
| Full method (content-space keys) | 0.899 | 0.090 | 87.1 % |
| − inference RoPE (content fit, no re-rotation) | 0.762 | 0.753 | 69.3 % |
| − all RoPE (fit and apply on rotated keys) | 0.871 | 0.220 | 83.2 % |
| − RoPE − cross-layer (rotated keys, k = 1) | 0.791 | 0.363 | 77.6 % |
| − RoPE − cross-layer − ridge (λ = 0) | 0.791 | 0.363 | 77.3 % |

Every component the paper credits is needed here. Skipping the re-rotation is worse than not
transferring at all (69.3 % against 75.0 %). Fitting on rotated keys more than doubles the KL, a
larger gap than the paper describes at this context length. The ridge penalty makes no measurable
difference with 128,000 samples per head.

## Latency: mapper against target re-prefill (k = 12)

Target transformer body without the LM head, as in the paper. 5 warmup + 10 timed trials per length
(the paper uses 50 + 30) and a cap at 8,192 tokens, both for thermal reasons.

| Tokens | Mapper | Re-prefill | Speedup |
|---:|---:|---:|---:|
| 64 | 27.9 ms | 78.4 ms | 2.8× |
| 256 | 48.3 ms | 100.8 ms | 2.1× |
| 1,024 | 148.5 ms | 261.7 ms | 1.8× |
| 2,048 | 300.7 ms | 588.0 ms | 2.0× |
| 4,096 | 618.9 ms | 1,211.2 ms | 2.0× |
| 8,192 | 1,249.3 ms | 2,447.2 ms | 2.0× |

This is below the paper's 2.7–25× range. Two reasons are specific to the pair: an 8B target is
cheap to re-prefill, and mapper cost grows with k (twelve source layers feed each target layer).
The paper's large ratios come from a 32B target at k = 8. Untested ways to widen the gap here: a
smaller k (in the smoke run a k = 1 mapper took 63 ms at 2,048 tokens), and running the mapper in
bf16 instead of fp32. The per-request energy figures in `bench.json` include governor pauses and
both operations together, so they are not a usable energy comparison.

## Other directions

* **Large → small** (8B → 4B, k = 12): attention cosine 0.878 (worst layer 0.525), KL 0.081,
  top-1 87.4 %. Handing a conversation back to the small model is viable on this pair.
* **Multi-turn**: ten 64-token turns alternating the live model every turn, mapping the whole cache
  at each switch. KL against the target's standalone distribution stays between 0.07 and 0.15 with
  no trend (slope −0.002 per turn). The paper measures this with CoQA F1; this is a logit-level
  analogue.

## What this does and does not establish

Established on this hardware:

* The implementation reproduces the paper's qualitative findings on a real matched-KV pair: high
  retention, a moderate best k, the RoPE ablation ordering, a working reverse direction.
* The full protocol fits and completes on the Spark under the thermal governor (about four hours,
  peak 81 °C, 26 GiB minimum available memory).

Not established:

* That transfer preserves a larger model's accuracy advantage. Source and target are level on two
  of the three tasks here.
* The paper's speedups. This pair gives 2×.
* Anything about the pod's real escalation pair, Qwen2.5-7B → 14B, which has mismatched KV heads
  (4 against 8) and is outside what the paper tested.

Limits of the measurement: 500 documents per task rather than full sets; three of the paper's five
benchmarks (no MMLU, no GSM8K); k chosen by attention cosine rather than by benchmark accuracy as in
the paper; a single calibration run and seed.

## Next steps

1. Tier 2: Qwen2.5-7B → Qwen2.5-14B with `--allow-mismatched`. It needs the bf16 Hugging Face
   checkpoints (about 45 GB of downloads); the pod's AWQ builds cannot be used. This is the pair
   that decides whether the idea helps the pod.
2. On this pair: rerun the harness at k = 4 and k = 8 to see what the cheaper mappers cost in
   accuracy, and benchmark a bf16 mapper.
3. A matched-KV pair with a real accuracy gap (Qwen3-4B → Qwen3-14B fits on the Spark) to test
   whether the target's advantage survives the transfer.
