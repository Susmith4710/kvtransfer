import numpy as np
import pytest
import torch

from kvtransfer import CrossModelTransfer, Mapper, Session, benchmark, calibrate, evaluate, format_rows
from kvtransfer.metrics import cache_r2

from conftest import random_batches, tiny_llama


@pytest.fixture(scope="module")
def pair(src_model, tgt_model):
    stats = calibrate(src_model, tgt_model, random_batches(6, 4, 48, seed=3), stride=2)
    return Mapper.fit(stats, k=2, lam=0.01)


@pytest.fixture(scope="module")
def ident_mapper(src_model):
    stats = calibrate(src_model, src_model, random_batches(8, 4, 64, seed=7), stride=1)
    return Mapper.fit(stats, k=1, lam=1e-6)


def test_generate_runs_and_extends_prompt(src_model, tgt_model, pair):
    xfer = CrossModelTransfer(src_model, tgt_model, pair)
    ids = torch.randint(0, 257, (2, 12))
    res = xfer.generate(ids, max_new_tokens=5, hold_back=2)
    assert res.tokens.shape == (2, 17)
    assert torch.equal(res.tokens[:, :12], ids)
    assert res.n_mapped == 10
    assert res.target_cache.get_seq_length() == 17  # the cache covers every returned token


def test_identity_transfer_matches_target_generate(src_model, ident_mapper):
    xfer = CrossModelTransfer(src_model, src_model, ident_mapper)
    ids = torch.randint(0, 257, (1, 20))
    res = xfer.generate(ids, max_new_tokens=6, hold_back=1)
    ref = src_model.generate(ids, max_new_tokens=6, do_sample=False)
    assert torch.equal(res.tokens, ref)


def test_geometry_check_rejects_wrong_model(src_model, tgt_model, pair):
    with pytest.raises(ValueError, match="geometry"):
        CrossModelTransfer(tiny_llama(2, seed=4), tgt_model, pair)      # wrong depth
    with pytest.raises(ValueError, match="geometry"):
        CrossModelTransfer(src_model, src_model, pair)                  # wrong target


def test_session_multi_turn_switching(src_model, tgt_model, pair):
    # a reverse mapper so we can go back
    stats_back = calibrate(tgt_model, src_model, random_batches(6, 4, 48, seed=4), stride=2)
    back = Mapper.fit(stats_back, k=2, lam=0.01)
    sess = Session({"s": src_model, "t": tgt_model}, {("s", "t"): pair, ("t", "s"): back}, start="s")
    sess.feed(torch.randint(0, 257, (1, 10)))
    a = sess.generate(3)
    assert a.shape == (1, 3) and sess.tokens.shape[1] == 13
    sess.switch_to("t")
    assert sess.live == "t" and sess.cache.get_seq_length() == 13
    sess.feed(torch.randint(0, 257, (1, 4)))
    b = sess.generate(2)
    assert b.shape == (1, 2) and sess.tokens.shape[1] == 19
    sess.switch_to("s")
    assert sess.live == "s" and sess.cache.get_seq_length() == 19
    c = sess.generate(2)
    assert sess.tokens.shape[1] == 21


def test_evaluate_identity_is_perfect(src_model, ident_mapper):
    seqs = [torch.randint(0, 257, (1, 40)) for _ in range(3)]
    rep = evaluate(src_model, src_model, ident_mapper, seqs, prefix_len=32, suffix_len=8)
    assert rep.n_sequences == 3
    assert np.mean(rep.r2_K) > 0.999 and np.mean(rep.r2_V) > 0.999
    assert rep.attn_cosine_mean > 0.999
    assert rep.kl_mean < 1e-3
    assert rep.top1_agreement == 1.0
    assert "attn cosine" in rep.summary()
    assert rep.to_dict()["n_sequences"] == 3


def test_evaluate_cross_model_runs(src_model, tgt_model, pair):
    seqs = [torch.randint(0, 257, (1, 30)) for _ in range(2)]
    rep = evaluate(src_model, tgt_model, pair, seqs, prefix_len=24, suffix_len=6)
    assert rep.n_sequences == 2 and len(rep.attn_cosine_layers) == 4
    assert np.isfinite(rep.kl_mean)
    r2 = cache_r2(pair, src_model, tgt_model, torch.randint(0, 257, (1, 20)))
    assert len(r2["K"]) == 4


def test_benchmark_runs(src_model, tgt_model, pair):
    rows = benchmark(src_model, tgt_model, pair, seq_lens=(8, 16), warmup=1, trials=2)
    assert [r.seq_len for r in rows] == [8, 16]
    assert all(r.mapper_ms > 0 and r.reprefill_ms > 0 for r in rows)
    assert "speedup" in format_rows(rows)


def test_llama_family_pipeline():
    """Different family code path (LlamaForCausalLM) end to end."""
    s, t = tiny_llama(2, seed=11), tiny_llama(3, seed=12)
    stats = calibrate(s, t, random_batches(4, 4, 32, seed=5), stride=2)
    m = Mapper.fit(stats, k="all", lam=0.01)
    res = CrossModelTransfer(s, t, m).generate(torch.randint(0, 257, (1, 9)), max_new_tokens=3)
    assert res.tokens.shape == (1, 12)
