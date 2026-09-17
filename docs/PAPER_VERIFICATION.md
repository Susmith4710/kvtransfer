# Paper verification: claim → code → test

Paper: Heo et al., "Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping
for Prefill Reuse", arXiv:2608.03893 (NVIDIA, Aug 2026). The paper released no code; everything
here is reconstructed from its equations and protocol descriptions. Tests run offline on CPU with
tiny random-init Qwen3 / Llama models, so they verify the mathematics and plumbing, not the
paper's accuracy numbers, which require the real checkpoints (see `docs/DGX_SPARK.md`).

| # | Paper | Code | Test |
|---|---|---|---|
| 1 | §2.1 matched-KV definition: same KV head count and per-head dim; depth may differ | `hf.check_matched_kv` | `test_mapper.py::test_model_spec_and_matched_kv`, `test_mismatched_kv_is_refused` |
| 2 | §2.1 shared tokenizer so token positions align | `hf.tokenizer_compatibility`, `assert_shared_tokenizer` | `test_extensions.py::test_tokenizer_compatibility_levels` |
| 3 | §2.3 / Eq. 2 single-source OLS probe per (l', l, h), head-matched, R² head-averaged, for K_rope, K_stripped, V | `select.probe_r2` (lam=0), `calibration` kinds `K`, `V`, `Krope` | `test_mapper.py::test_probe_r2_diagonal_for_identity` (diagonal R²=1 on an identity pair), `test_identity_pair_reproduces_native_logits` (argmax of the score is the diagonal) |
| 4 | §3.2 top-k source layers per target layer by head-averaged R² averaged over K_stripped and V; selection shared across heads | `select.selection_score(kinds=("K","V"))["mean"]`, `select.top_k_layers`, `Mapper.fit` | `test_mapper.py::test_calibrate_fit_apply_shapes` |
| 5 | §3.1 Eq. 3: X = concatenation of all KV heads of the k selected layers; W ∈ ℝ^{(k·n_kv·d_h)×d_h} per head | `calibration.CalibrationStats.src_rows`, `Mapper.fit` solves all target heads of a layer at once (columns independent ⇒ identical to per-head) | `test_mapper.py::test_calibrate_fit_apply_shapes` (W shape), `test_ridge.py::test_block_solve_matches_full_solve_on_subblock` |
| 6 | §3.1 Eq. 4: W* = (XᵀX + λI)⁻¹XᵀY on centered X, Y; b = Ȳ − X̄W*; λ = 0.01 | `ridge.MomentAccumulator` (shifted moments, centered Gram/Cross, `solve`) | `test_ridge.py::test_streaming_ridge_recovers_planted_affine_map`, `test_state_dict_round_trip`, `test_r2_of_pure_noise_is_near_zero` |
| 7 | §3.1 fitting cost shared across heads: XᵀX formed once per target layer | one `solve` per (layer, kind) with all target-head columns | same as 5 |
| 8 | §3.3 content-space mapping: strip source RoPE, map, re-apply target RoPE; Y = target's RoPE-stripped keys; values direct; inversion exact because R is orthogonal | `rope.RopeCodec` (cos/sin from the model's own rotary module, `attention_scaling` handled), `calibration.extract_content_kv`, `Mapper.apply_kv` | `test_rope.py::*` (exact inverse; bitwise match with HF `apply_rotary_pos_emb`; YaRN scaling; cache key == RoPE(post-norm key)) |
| 9 | §3.3 position-free fit reusable at other context lengths | same | `test_mapper.py::test_content_space_mapper_generalizes_past_calibration_length` (fit at 32 tokens, exact logits at 200) |
| 10 | §3.1 calibration: 500 × 1,024 FineWeb-Edu, stride 4 ⇒ ~128K tokens per head; bf16 forward, fp32 covariance, fp64 solve | `ExperimentConfig` / `cli fit` defaults; `calibrate(stats_dtype=float32)`, `solve(solve_dtype=float64)`; `data.iter_fineweb_edu` | `test_hardware_catalog.py::test_dgx_spark_plan_for_paper_pair` (128,000 tokens per head) |
| 11 | App. D size formula 2·L_t·n_kv·(k·n_kv·d_h)·d_h and Table 12 sizes (1.07 B for Qwen3 14B→32B k=8, etc.) | `Mapper.formula_params`, `PairPlan.mapper_bytes` | `test_hardware_catalog.py::test_appendix_d_parameter_counts` (all six paper pairs within 1 %) |
| 12 | Table 2 ablations: −inference RoPE (content fit, no re-rotation), −all RoPE (fit+apply on rotated keys), −cross-layer (k=1) | `Mapper.ablate_inference_rope()`, `Mapper.fit(key_space="rope")`, `experiment` stage `ablation` | `test_extensions.py::test_rope_space_ablation_variants` (rope-space ≈ full at calibration length; no-re-rotation clearly worse) |
| 13 | §4.5 attention-output cosine between mapped-KV and ground-truth-KV attention outputs, averaged over layers and heads | `metrics._AttnOutputTap` (input of `o_proj`), `metrics.evaluate` | `test_transfer.py::test_evaluate_identity_is_perfect` (cosine → 1, KL → 0) |
| 14 | §4.5 R² does not predict retention; cosine does | `experiment` picks best k by cosine and says so in `report.md` | design choice, documented |
| 15 | §4.6 multi-turn handoff alternating models across turns | `transfer.Session`, `experiment.multiturn_drift` | `test_transfer.py::test_session_multi_turn_switching`, `test_extensions.py::test_experiment_end_to_end_on_tiny_models` |
| 16 | §4.7 / App. G latency: re-prefill = target transformer body without LM head; mapper eager; warmup + timed trials; 64…32,768 tokens; cross-device shipment included | `bench.benchmark` (`model.model`, warmup/trials, copies to the target device when different), `experiment` stage `bench` | `test_transfer.py::test_benchmark_runs` |
| 17 | §4.1 retention = transfer/target; floor-normalized = (acc − chance)/(target − chance); chance 25/25/50/25/≈0 | `lm_eval_adapter.retention_table`, `CHANCE` | `test_lm_eval_adapter.py` retention tests |
| 18 | §4.1 benchmarks via lm-evaluation-harness defaults (log-likelihood scoring; MMLU 5-shot; GSM8K 8-shot CoT generation) | `lm_eval_adapter.TransferLM` registered as `--model kvtransfer` (loglikelihood + generate_until) | `test_lm_eval_adapter.py` (matches stock `HFLM` on an identity pair) |
| 19 | Fig. 1 pipeline: source prefill → mapped cache → target decodes | `transfer.CrossModelTransfer.generate`; last `hold_back` prompt tokens run on the target (the paper leaves this implicit) | `test_transfer.py::test_identity_transfer_matches_target_generate` (token-exact vs `model.generate`) |
| 20 | Sec. 4.2 both directions (S→L and L→S) | any pair order; `Session` with two mappers | `test_transfer.py::test_session_multi_turn_switching` |

## Deviations and extensions (documented)

* **Best-k selection.** The paper selects k per pair by benchmark accuracy (App. H). `experiment`
  selects k by mean attention-output cosine on held-out text because it needs no benchmarks;
  `kvtransfer harness` gives the paper's criterion.
* **Layer probe uses ridge moments.** The paper's probe is OLS on stride-subsampled tokens; ours is
  the same OLS (λ = 0) computed from the streamed moments, so it needs no token storage. Identical
  in exact arithmetic.
* **Mismatched KV.** The paper never tests different KV head counts or head dims. With
  `--allow-mismatched` the probe regresses each target head on all source heads of a layer, and
  the ridge is the same equation with a different feature width. This is an extension.
* **Cross-family with a shared tokenizer** (Qwen2.5 ↔ Qwen3) is likewise beyond the paper.
* **Not implemented.** The nonlinear MLP mapper (§4.4), greedy forward selection (App. B),
  the λ / N / calibration-domain sweeps (App. C, though every knob is exposed), CoQA F1 itself
  (the multi-turn stage measures KL and top-1 agreement against the target's standalone
  distribution instead).

## Things the tests cannot verify

Accuracy retention, the 73–98 % / 42 % tier split, the 2.7–25× latency ratios, and the
attention-cosine/retention correlation all need the real checkpoints and hardware. The Spark
runbook produces exactly those numbers; the tests only guarantee that the pipeline reproduces a
model's own logits when the mapper should be the identity, at and beyond the calibration length.
