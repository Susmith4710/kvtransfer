"""Mismatched-KV and cross-family paths, RoPE ablations, tokenizer compatibility, energy meter, serve, experiment."""
import json
import threading
import urllib.request

import numpy as np
import pytest
import torch

from kvtransfer import CrossModelTransfer, Mapper, calibrate, evaluate
from kvtransfer.energy import EnergyMeter
from kvtransfer.hf import tokenizer_compatibility
from kvtransfer.select import probe_r2, selection_score

from conftest import random_batches, tiny_qwen3


def test_mismatched_kv_heads_pipeline_runs_with_warning(src_model):
    other = tiny_qwen3(4, seed=21, n_kv=4)                       # 2 KV heads -> 4 KV heads
    with pytest.warns(UserWarning, match="mismatched-KV"):
        stats = calibrate(src_model, other, random_batches(6, 4, 48, seed=3), stride=2, require_matched_kv=False)
    r2 = probe_r2(stats, "K")
    assert r2.shape == (3, 4) and np.isfinite(r2).all()
    m = Mapper.fit(stats, k=2, lam=0.01)
    assert m.W_K[0].shape == (2 * 2 * 16, 4 * 16)
    res = CrossModelTransfer(src_model, other, m).generate(torch.randint(0, 257, (1, 10)), max_new_tokens=3)
    assert res.tokens.shape == (1, 13)
    rep = evaluate(src_model, other, m, [torch.randint(0, 257, (1, 30))], prefix_len=24, suffix_len=6)
    assert np.isfinite(rep.attn_cosine_mean)


def test_mismatched_head_dim_pipeline_runs(src_model):
    other = tiny_qwen3(3, seed=22, n_kv=2, head_dim=8)           # 16 -> 8 head dim
    stats = calibrate(src_model, other, random_batches(4, 4, 32, seed=5), stride=2, require_matched_kv=False)
    m = Mapper.fit(stats, k="all", lam=0.01)
    assert m.W_V[0].shape == (3 * 2 * 16, 2 * 8)
    res = CrossModelTransfer(src_model, other, m).generate(torch.randint(0, 257, (1, 8)), max_new_tokens=2)
    assert res.tokens.shape == (1, 10)


def test_rope_space_ablation_variants(src_model):
    """Identity pair: content-space and rope-space fits both reproduce logits at the calibration length
    (paper: '-all RoPE' within noise), while '-inference RoPE' (no re-rotation) must be clearly worse."""
    batches = random_batches(8, 4, 64, seed=9)
    stats = calibrate(src_model, src_model, batches, stride=1, kinds=("K", "V", "Krope"))
    assert set(stats.acc) == {"K", "V", "Krope"}
    score = selection_score(stats)["mean"]
    full = Mapper.fit(stats, k=1, lam=1e-6, score=score)
    rope = Mapper.fit(stats, k=1, lam=1e-6, score=score, key_space="rope")
    noinf = full.ablate_inference_rope()
    assert rope.key_space == "rope" and noinf.key_space == "content-norerotate"
    seqs = [torch.randint(0, 257, (1, 48)) for _ in range(2)]
    r_full = evaluate(src_model, src_model, full, seqs, prefix_len=40, suffix_len=8)
    r_rope = evaluate(src_model, src_model, rope, seqs, prefix_len=40, suffix_len=8)
    r_noinf = evaluate(src_model, src_model, noinf, seqs, prefix_len=40, suffix_len=8)
    assert r_full.kl_mean < 1e-3 and r_rope.kl_mean < 1e-3
    assert r_noinf.kl_mean > 10 * max(r_full.kl_mean, 1e-6)
    # rope-space mapper survives save/load with its key_space
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        rope.save(d)
        assert Mapper.load(d).key_space == "rope"


def test_tokenizer_compatibility_levels():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    sents = ["the quick brown fox jumps", "a b c d e f g", "keys and values"] * 30

    def make(extra_special):
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tok.train_from_iterator(sents, trainers.BpeTrainer(vocab_size=120, special_tokens=["<unk>"] + extra_special))
        return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>")

    a, b = make([]), make([])
    assert tokenizer_compatibility(a, b) == "identical"
    c = make(["<think>"])       # same BPE, extra special token appended -> different vocab, same ids on text
    level = tokenizer_compatibility(a, c, texts=("the quick brown fox jumps",))
    assert level in ("compatible", "incompatible")


def test_energy_meter_is_safe_without_nvml():
    with EnergyMeter() as m:
        sum(range(1000))
    assert m.reading is not None and m.reading.seconds >= 0
    assert m.reading.method in ("none", "nvml_energy_counter", "nvml_power_sampling")
    d = m.reading.to_dict()
    assert "joules" in d


@pytest.fixture(scope="module")
def pair_mapper(src_model, tgt_model):
    stats = calibrate(src_model, tgt_model, random_batches(6, 4, 48, seed=3), stride=2)
    return Mapper.fit(stats, k=2, lam=0.01)


def test_escalator_shares_prefix_and_reports_skipped_tokens(src_model, tgt_model, pair_mapper):
    from kvtransfer.serve import Escalator, common_prefix_len
    assert common_prefix_len(torch.tensor([[1, 2, 3, 4]]), torch.tensor([[1, 2, 9, 4, 5]])) == 2
    assert common_prefix_len(torch.tensor([[1, 2]]), torch.tensor([[1, 2, 3]])) == 2

    class Tok:  # minimal tokenizer stand-in
        eos_token_id = None

        def __call__(self, text, return_tensors=None):
            ids = [ord(c) % 250 for c in text]
            return {"input_ids": torch.tensor([ids])}

        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(97 + (i % 26)) for i in ids)

    esc = Escalator(src_model, tgt_model, pair_mapper, Tok(), chat_default=False)
    r1 = esc.generate({"session": "s1", "role": "source", "prompt": "SYSTEM: briefing about cats. Q: why?", "max_new_tokens": 4})
    assert r1["new_tokens"] == 4 and r1["timing"]["prefill_tokens"] > 0
    r2 = esc.generate({"session": "s1", "role": "escalate", "prompt": "SYSTEM: briefing about cats. Q: explain", "max_new_tokens": 3})
    assert r2["timing"]["mode"] == "transfer"
    assert r2["timing"]["skipped_tokens"] == len("SYSTEM: briefing about cats. Q: ") - 0 or r2["timing"]["skipped_tokens"] > 0
    assert r2["timing"]["tail_tokens"] >= 1 and "mapper_ms" in r2["timing"]
    r3 = esc.generate({"session": "s1", "role": "escalate", "prompt": "SYSTEM: briefing about cats. Q: explain", "baseline": True, "max_new_tokens": 3})
    assert r3["timing"]["mode"] == "re-prefill baseline" and r3["timing"]["skipped_tokens"] == 0
    r4 = esc.generate({"session": "new", "role": "escalate", "prompt": "anything", "max_new_tokens": 2})
    assert "no source cache" in r4["timing"]["mode"]
    r5 = esc.generate({"session": "s1", "role": "target", "prompt": "anything", "max_new_tokens": 2})
    assert r5["role"] == "target"
    esc.reset("s1")
    assert "s1" not in esc.sessions


def test_http_server_round_trip(src_model, tgt_model, pair_mapper):
    from kvtransfer.serve import Escalator, serve

    class Tok:
        eos_token_id = None

        def __call__(self, text, return_tensors=None):
            return {"input_ids": torch.tensor([[ord(c) % 250 for c in text]])}

        def decode(self, ids, skip_special_tokens=True):
            return "x" * len(ids)

    srv = serve(Escalator(src_model, tgt_model, pair_mapper, Tok(), chat_default=False), "127.0.0.1", 0)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as r:
            assert json.loads(r.read())["ok"]
        body = json.dumps({"session": "a", "role": "source", "prompt": "hello world", "max_new_tokens": 2}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/generate", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            out = json.loads(r.read())
        assert out["role"] == "source" and out["new_tokens"] == 2
        body = json.dumps({"session": "a", "role": "escalate", "prompt": "hello world!", "max_new_tokens": 2}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/generate", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            out = json.loads(r.read())
        assert out["timing"]["mode"] == "transfer" and out["timing"]["skipped_tokens"] > 0
    finally:
        srv.shutdown()


def test_experiment_end_to_end_on_tiny_models(tmp_path):
    """Whole protocol on tiny models with a local text file, all stages, resumable."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    from kvtransfer.experiment import ExperimentConfig, run_experiment

    sents = ["the quick brown fox jumps over the lazy dog", "a small model prefills and a large model decodes",
             "keys are rotated by position and values are not", "ridge regression is a closed form linear solve"] * 40
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(sents, trainers.BpeTrainer(vocab_size=257, special_tokens=["<unk>", "<eos>"]))
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", eos_token="<eos>")
    paths = {}
    for name, m in (("src", tiny_qwen3(3, seed=1)), ("tgt", tiny_qwen3(4, seed=2))):
        m.save_pretrained(tmp_path / name)
        hf_tok.save_pretrained(tmp_path / name)
        paths[name] = str(tmp_path / name)
    data = tmp_path / "calib.txt"
    docs = [" ".join(sents[i:i + 40]) for i in range(0, len(sents) - 40, 2)]
    data.write_text("\n\n".join(docs))
    cfg = ExperimentConfig(paths["src"], paths["tgt"], str(tmp_path / "exp"), data=str(data), n_seqs=8, seq_len=32, stride=2,
                           batch_size=4, k_values=(1, 2, "all"), eval_n_seqs=2, eval_seq_len=32, eval_suffix_len=4,
                           bench_seq_lens=(8, 16), bench_warmup=1, bench_trials=1, multiturn_turns=4, multiturn_turn_tokens=8,
                           device="cpu", dtype="float32", hardware="dgx-spark", force=True)
    rep = run_experiment(cfg, log=lambda *a: None)
    out = tmp_path / "exp"
    assert (out / "report.md").exists() and (out / "stats" / "meta.json").exists()
    assert sorted(p.name for p in (out / "mappers").iterdir()) == ["k1", "k2", "k3"]
    assert rep["best_k"] in (1, 2, 3)
    assert set(rep["stages"]) >= {"plan", "calibrate", "fit", "eval", "ablation", "bench", "multiturn"}
    assert len(rep["stages"]["ablation"]) == 5
    assert rep["stages"]["multiturn"]["alternating"] is True
    assert [r["live"] for r in rep["stages"]["multiturn"]["turns"]] == ["s", "t", "s", "t"]
    assert (out / "reverse" / "mapper" / "mapper.json").exists() and rep["stages"]["reverse"]["k"] >= 1
    # resume: second run must reuse everything and not recalibrate
    rep2 = run_experiment(cfg, log=lambda *a: None)
    assert "calibrate" not in rep2["stages"]  # reused stats, no new calibration timing
    md = (out / "report.md").read_text()
    assert "Ablation" in md and "Latency" in md and "Multi-turn" in md


def test_generate_cache_covers_all_tokens_even_with_eos(src_model, tgt_model, pair_mapper):
    """Review finding: an EOS stop must not leave the cache one token short of the token sequence."""
    xfer = CrossModelTransfer(src_model, tgt_model, pair_mapper)
    ids = torch.randint(0, 257, (1, 9))
    # find the greedy first token and use it as EOS so generation stops immediately
    logits, _ = xfer.handoff(ids, hold_back=1)
    eos = int(logits[:, -1].argmax(-1))
    res = xfer.generate(ids, max_new_tokens=5, hold_back=1, eos_token_id=eos)
    assert res.tokens.shape[1] == 10 and int(res.tokens[0, -1]) == eos
    assert res.target_cache.get_seq_length() == res.tokens.shape[1]
    # Session: same invariant, and feeding after EOS keeps positions aligned
    from kvtransfer import Session
    sess = Session({"s": src_model, "t": tgt_model}, {("s", "t"): pair_mapper}, start="t")
    sess.feed(ids)
    eos2 = int(sess.last_logits.argmax(-1))
    sess.generate(5, eos_token_id=eos2)
    assert sess.cache.get_seq_length() == sess.tokens.shape[1]
    sess.feed(torch.randint(0, 257, (1, 3)))
    assert sess.cache.get_seq_length() == sess.tokens.shape[1]


def test_escalator_cache_alignment_with_eos(src_model, tgt_model, pair_mapper):
    from kvtransfer.serve import Escalator

    class Tok:
        eos_token_id = 7

        def __call__(self, text, return_tensors=None):
            return {"input_ids": torch.tensor([[ord(c) % 250 for c in text]])}

        def decode(self, ids, skip_special_tokens=True):
            return "x" * len(ids)

    esc = Escalator(src_model, tgt_model, pair_mapper, Tok(), chat_default=False)
    esc.generate({"session": "e", "role": "source", "prompt": "hello there world", "max_new_tokens": 6})
    st = esc.sessions["e"]
    assert st.source_cache.get_seq_length() == st.source_tokens.shape[1]
    r = esc.generate({"session": "e", "role": "escalate", "prompt": "hello there world again", "max_new_tokens": 2})
    assert r["timing"]["skipped_tokens"] <= st.source_cache.get_seq_length()


def test_paper_notes_are_direction_specific():
    from kvtransfer.catalog import BY_ID, classify_pair
    small, large = BY_ID["meta-llama/Llama-3.1-8B-Instruct"], BY_ID["meta-llama/Llama-3.1-70B-Instruct"]
    # 70B is flagged unsupported on a 128 GB box, so test the note lookup directly
    from kvtransfer.discover import PAPER_PAIRS
    assert "Tier 1" in PAPER_PAIRS[("Llama-3.1-8B", "Llama-3.1-70B")]
    assert "37 %" in PAPER_PAIRS[("Llama-3.1-70B", "Llama-3.1-8B")]
    q14, q32 = BY_ID["Qwen/Qwen3-14B"], BY_ID["Qwen/Qwen3-32B"]
    assert classify_pair(q14, q32)[0] == "paper-validated" and "97.6" in classify_pair(q14, q32)[1]
    assert classify_pair(q32, q14)[0] == "paper-validated" and "L->S" in classify_pair(q32, q14)[1]


def test_selection_score_raises_on_missing_kind(src_model):
    stats = calibrate(src_model, src_model, random_batches(2, 2, 16, seed=1), stride=1, kinds=("V",))
    with pytest.raises(ValueError, match="lack kind"):
        selection_score(stats)


def test_discover_tolerates_file_roots(tmp_path):
    from kvtransfer.discover import scan
    f = tmp_path / "not_a_dir.txt"
    f.write_text("x")
    assert scan([f], include_hf_cache=False) == []


def test_retention_summary():
    from kvtransfer.lm_eval_adapter import retention_summary
    rows = [{"retention_pct": 90.0, "normalized_retention_pct": 80.0}, {"retention_pct": 100.0, "normalized_retention_pct": None}]
    s = retention_summary(rows)
    assert s["n_tasks"] == 2 and s["avg_retention_pct"] == 95.0 and s["avg_floor_normalized_pct"] == 80.0


def test_fit_r2_is_head_averaged(src_model):
    """Head-averaged R^2 must equal the mean of per-head solves (paper Table 7 / App. B)."""
    stats = calibrate(src_model, src_model, random_batches(4, 4, 32, seed=2), stride=1)
    m = Mapper.fit(stats, k=1, lam=0.01)
    lt = 1
    rows = stats.src_rows(m.selected[lt].tolist())
    per_head = [stats.acc["K"].solve(0.01, rows=rows, cols=stats.tgt_cols(lt, h))[2] for h in range(stats.target.n_kv)]
    assert abs(m.fit_r2["K"][lt] - float(np.mean(per_head))) < 1e-6
