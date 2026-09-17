import numpy as np
import torch

from kvtransfer import CalibrationStats, Mapper, calibrate, model_spec, probe_r2, selection_score, top_k_layers
from kvtransfer.hf import cache_layer, cache_to_list, forward_with_cache, list_to_cache
from kvtransfer.rope import RopeCodec

from conftest import random_batches


def test_model_spec_and_matched_kv(src_model, tgt_model):
    s, t = model_spec(src_model, "src"), model_spec(tgt_model, "tgt")
    assert (s.n_layers, s.n_kv, s.head_dim) == (3, 2, 16)
    assert (t.n_layers, t.n_kv, t.head_dim) == (4, 2, 16)
    from kvtransfer import check_matched_kv
    check_matched_kv(s, t)


def test_calibrate_fit_apply_shapes(src_model, tgt_model, calib_batches):
    stats = calibrate(src_model, tgt_model, calib_batches, stride=2)
    assert stats.acc["K"].n == 6 * 4 * 24
    score = selection_score(stats)
    assert score["mean"].shape == (3, 4)
    sel = top_k_layers(score["mean"], 2)
    assert sel.shape == (4, 2)
    m = Mapper.fit(stats, k=2, lam=0.01, score=score["mean"])
    assert m.k == 2 and len(m.W_K) == 4 and m.W_K[0].shape == (2 * 32, 32)
    assert m.n_params(with_bias=False) == Mapper.formula_params(m.target, m.source, 2)
    # apply
    ids = torch.randint(0, 257, (2, 17))
    out = src_model(input_ids=ids, use_cache=True)
    kvs = cache_to_list(out.past_key_values)
    mapped = m.apply_kv(kvs, torch.arange(17), RopeCodec.from_model(src_model), RopeCodec.from_model(tgt_model))
    assert len(mapped) == 4 and mapped[0][0].shape == (2, 2, 17, 16)
    cache = list_to_cache(mapped, tgt_model)
    o = forward_with_cache(tgt_model, cache, torch.randint(0, 257, (2, 3)), past_len=17)
    assert o.logits.shape == (2, 3, 257)


def test_identity_pair_reproduces_native_logits(src_model):
    """Source == target: the ridge map must be ~identity and the target's logits from the mapped cache
    must equal the logits from its own cache.  This validates injection, RoPE handling, and centering."""
    batches = random_batches(n_batches=8, batch=4, T=64, seed=7)
    stats = calibrate(src_model, src_model, batches, stride=1)
    score = selection_score(stats)
    # the most predictive source layer for each target layer is itself
    assert np.array_equal(np.argmax(score["mean"], axis=0), np.arange(3))
    m = Mapper.fit(stats, k=1, lam=1e-6, score=score["mean"])
    assert np.mean(m.fit_r2["K"]) > 0.999 and np.mean(m.fit_r2["V"]) > 0.999
    ids = torch.randint(0, 257, (1, 40))
    out = src_model(input_ids=ids, use_cache=True)
    codec = RopeCodec.from_model(src_model)
    mapped = m.apply_kv([(k[:, :, :30], v[:, :, :30]) for k, v in cache_to_list(out.past_key_values)],
                        torch.arange(30), codec, codec)
    native = list_to_cache([(k[:, :, :30], v[:, :, :30]) for k, v in cache_to_list(out.past_key_values)], src_model)
    tail = ids[:, 30:]
    lo_native = forward_with_cache(src_model, native, tail, past_len=30).logits
    lo_mapped = forward_with_cache(src_model, list_to_cache(mapped, src_model), tail, past_len=30).logits
    assert torch.allclose(lo_native, out.logits[:, 30:], atol=1e-4)  # injection path is exact
    assert torch.allclose(lo_mapped, lo_native, atol=2e-2)
    assert torch.equal(lo_mapped.argmax(-1), lo_native.argmax(-1))


def test_probe_r2_diagonal_for_identity(src_model, calib_batches):
    stats = calibrate(src_model, src_model, calib_batches, stride=1)
    r2 = probe_r2(stats, "K")
    assert np.allclose(np.diag(r2), 1.0, atol=1e-4)


def test_save_load_round_trip(tmp_path, src_model, tgt_model, calib_batches):
    stats = calibrate(src_model, tgt_model, calib_batches, stride=2)
    stats.save(tmp_path / "stats")
    stats2 = CalibrationStats.load(tmp_path / "stats")
    m1 = Mapper.fit(stats, k="all", lam=0.01)
    m2 = Mapper.fit(stats2, k="all", lam=0.01)
    assert all(torch.allclose(a, b) for a, b in zip(m1.W_K, m2.W_K))
    m1.save(tmp_path / "mapper")
    m3 = Mapper.load(tmp_path / "mapper")
    assert m3.k == 3 and np.array_equal(m3.selected, m1.selected)
    assert all(torch.equal(a, b) for a, b in zip(m1.W_V, m3.W_V))
    assert m3.fit_r2["K"] == m1.fit_r2["K"]
    m1.save(tmp_path / "mapper_bf16", dtype=torch.bfloat16)
    m4 = Mapper.load(tmp_path / "mapper_bf16")
    assert m4.W_K[0].dtype == torch.bfloat16
    assert "R2" in m4.summary()


def test_mismatched_kv_is_refused(src_model):
    from conftest import tiny_qwen3
    other = tiny_qwen3(2, seed=9, n_kv=4)
    import pytest
    with pytest.raises(ValueError, match="matched-KV"):
        calibrate(src_model, other, random_batches(1, 2, 16), stride=1)


def test_content_space_mapper_generalizes_past_calibration_length(src_model):
    """Fit on 32-token sequences, apply at 200 tokens: the position-free fit must still reproduce the
    target's logits (paper Sec. 3.3, the reason keys are mapped with RoPE stripped)."""
    stats = calibrate(src_model, src_model, random_batches(n_batches=10, batch=4, T=32, seed=11), stride=1)
    m = Mapper.fit(stats, k=1, lam=1e-6)
    ids = torch.randint(0, 257, (1, 210))
    out = src_model(input_ids=ids, use_cache=True)
    codec = RopeCodec.from_model(src_model)
    kvs = [(k[:, :, :200], v[:, :, :200]) for k, v in cache_to_list(out.past_key_values)]
    mapped = m.apply_kv(kvs, torch.arange(200), codec, codec)
    lo_mapped = forward_with_cache(src_model, list_to_cache(mapped, src_model), ids[:, 200:], past_len=200).logits
    assert torch.allclose(lo_mapped, out.logits[:, 200:], atol=2e-2)
    assert torch.equal(lo_mapped.argmax(-1), out.logits[:, 200:].argmax(-1))
