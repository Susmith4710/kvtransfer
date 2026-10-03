# Gap-pair results: Qwen3-1.7B → Qwen3-8B on the DGX Spark

Second real-model run (2026-10-02, one DGX Spark). Tier 1 used Qwen3-4B → Qwen3-8B, where source and
target scored about the same on the benchmarks, so it could not show whether the target's advantage
survives the transfer. This pair has a real gap: the source is 8 to 13 points weaker on every task.

Source `Qwen/Qwen3-1.7B` (28 layers), target `Qwen/Qwen3-8B` (36 layers), both 8 KV heads × 128,
identical tokenizers, bf16. Same protocol and settings as `docs/RESULTS_TIER1.md`: 500 × 1,024
FineWeb-Edu tokens, stride 4, λ = 0.01, 32 held-out sequences, first 500 documents per benchmark.
k sweep {1, 2, 4, 6, 8, 10, 12, 16, 20, all = 28}. Raw outputs: `runs/qwen3-1.7b-to-8b/`.

## Headline

**The transfer delivers roughly the small model's accuracy, not the large model's.** Decoding from the
mapped cache, the 8B recovers about a fifth of the gap between the 1.7B and itself.

| Task | Source alone | Target alone | Transfer (k = 10) | Retention | Share of the gap recovered |
|---|---:|---:|---:|---:|---:|
| ARC-Challenge (acc_norm) | 0.432 | 0.558 | 0.466 | 83.5 % | 27 % |
| HellaSwag (acc_norm) | 0.534 | 0.642 | 0.536 | 83.5 % | 2 % |
| WinoGrande (acc) | 0.604 | 0.684 | 0.630 | 92.1 % | 33 % |
| **Average** | | | | **86.4 %** | **20 %** |

Standard error is about ±2.2 points per cell, so the gains over the source on ARC-Challenge (+3.4)
and WinoGrande (+2.6) are within noise of zero, and HellaSwag (+0.2) is zero.

## Retention overstates the benefit

Retention, the paper's metric, divides transfer accuracy by target accuracy. On that scale this pair
scores 86.4 % (71.2 % floor-normalized), inside the paper's 73–98 % range for working pairs. But the
source model alone already reaches 83.0 % of the target's accuracy on these tasks. The transfer adds
3.4 points of retention on top of doing nothing. The informative number is the share of the
source-to-target gap that the transfer closes, and here it is about 20 %.

Tier 1 pointed the same way: on HellaSwag, its only task with a gap, the transfer landed at the
source's score (0.582 against 0.580).

## Why this may be a hard case

In these benchmarks the whole prompt except its last token comes from the mapped cache
(`hold_back = 1`), and the target only scores a short answer. Everything the large model knows
about the question arrives through the small model's representation. An escalation flow where the
large model also reads a tail of its own (the question and instruction after a shared system
prompt and briefing) gives it native context to work with, and may keep more of its advantage.
That is untested.

## Held-out diagnostics across k

Without any transfer the source's next-token choice matches the target's 59.4 % of the time
(75.0 % for the 4B source in Tier 1).

| k | In-sample R² K / V | Attention cosine | Worst layer | KL | Top-1 agreement |
|---|---:|---:|---:|---:|---:|
| 1 | 0.673 / 0.488 | 0.679 | 0.313 | 0.821 | 65.1 % |
| 2 | 0.718 / 0.556 | 0.798 | 0.540 | 0.553 | 70.6 % |
| 4 | 0.753 / 0.608 | 0.860 | 0.736 | 0.379 | 76.1 % |
| 6 | 0.771 / 0.637 | 0.869 | 0.697 | 0.311 | 78.2 % |
| 8 | 0.783 / 0.657 | 0.876 | 0.712 | 0.282 | 79.0 % |
| **10** | 0.793 / 0.673 | **0.880** | 0.722 | 0.261 | 80.3 % |
| 12 | 0.802 / 0.685 | 0.873 | 0.471 | 0.233 | 80.3 % |
| 16 | 0.815 / 0.706 | 0.862 | 0.488 | 0.227 | 81.7 % |
| 20 | 0.827 / 0.726 | 0.851 | 0.495 | **0.222** | **82.0 %** |
| 28 (all) | 0.847 / 0.760 | 0.694 | 0.367 | 0.326 | 77.1 % |

The mapped cache does move the target's next-token behaviour a long way toward its own (59 % to 80 %
agreement), and KL is about three times the 4B → 8B value (0.261 against 0.090). Cosine picks k = 10
while KL and top-1 keep improving to k = 20; the harness above used k = 10, so k = 16 or 20 may score
slightly better. Cross-layer sources matter much more than in Tier 1: k = 1 is far behind.

## Other measurements

* **Ablation at k = 10**: full method 0.880 / KL 0.261 / 80.3 %; no re-rotation 0.740 / 0.992 /
  63.2 %; fit on rotated keys 0.873 / 0.302 / 78.0 %; rotated keys with k = 1 0.659 / 0.908 /
  62.4 %; λ = 0 unchanged. Same ordering as Tier 1 and the paper.
* **Large → small** (8B → 1.7B, k = 10): cosine 0.840, KL 0.267, top-1 80.1 %.
* **Latency** (k = 10 mapper against 8B re-prefill): 3.2× at 64 tokens, 2.0–2.2× from 512 to 8,192
  tokens (8,192 tokens: 1,212 ms against 2,425 ms). The same roughly 2× as Tier 1.
* **Multi-turn**: KL between 0.15 and 0.25 over ten alternating handoffs, no trend
  (slope +0.002 per turn).
* **Run**: 2.9 hours governed, peak 79 °C, 60 GiB minimum available memory. The k = all fit took
  9 minutes with the slimmed shared solve.

## What this means

* On a pair with a real quality gap, a closed-form linear map of the small model's cache does not
  give the large model its usual accuracy on these tasks. It gives a result close to the small
  model's, for about half the large model's prefill time.
* That is weaker than the paper's framing suggests, and the reason is visible only when the source's
  own accuracy is reported next to the retention figure.
* Open questions worth one more run each: whether a longer natively processed tail (`--hold-back`
  16 or 32, or a prefix-only mapping as in the escalation server) recovers the gap, whether k = 16
  or 20 scores better than k = 10, and whether the paper's MLP mapper does better than ridge here.
