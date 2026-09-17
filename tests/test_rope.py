import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from kvtransfer.rope import RopeCodec, apply_rope, strip_rope

from conftest import tiny_qwen3


def test_strip_is_exact_inverse_of_apply():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 10, 16)
    freqs = torch.rand(10, 8) * 6.28          # rotate-half layout: each frequency appears twice
    emb = torch.cat([freqs, freqs], dim=-1)
    cos, sin = emb.cos(), emb.sin()
    y = apply_rope(x, cos, sin)
    assert torch.allclose(strip_rope(y, cos, sin), x, atol=1e-5)


def test_codec_matches_hf_apply_rotary_pos_emb(src_model):
    codec = RopeCodec.from_model(src_model)
    T, d = 20, src_model.config.head_dim
    pos = torch.arange(T)
    k = torch.randn(1, src_model.config.num_key_value_heads, T, d)
    cos, sin = src_model.model.rotary_emb(torch.zeros(1, 1), pos[None])
    q_dummy = torch.zeros(1, 4, T, d)
    _, k_hf = apply_rotary_pos_emb(q_dummy, k, cos, sin)
    k_ours = codec.apply(k, pos)
    assert torch.allclose(k_hf, k_ours, atol=1e-6)
    assert torch.allclose(codec.strip(k_ours, pos), k, atol=1e-5)


def test_codec_handles_attention_scaling():
    m = tiny_qwen3(2, seed=5, rope_scaling={"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 256})
    codec = RopeCodec.from_model(m)
    assert codec.attention_scaling != 1.0
    pos = torch.arange(30)
    k = torch.randn(1, 2, 30, 16)
    assert torch.allclose(codec.strip(codec.apply(k, pos), pos), k, atol=1e-5)


def test_cache_keys_are_rotated_content_keys(src_model):
    """What the model writes into its cache equals RoPE applied to the pre-rotation (post-norm) keys."""
    torch.manual_seed(0)
    ids = torch.randint(0, 257, (1, 12))
    captured = {}
    layer = src_model.model.layers[0].self_attn
    h = layer.k_norm.register_forward_hook(lambda m, i, o: captured.setdefault("k", o.detach()))
    out = src_model(input_ids=ids, use_cache=True)
    h.remove()
    from kvtransfer.hf import cache_layer
    k_cache, _ = cache_layer(out.past_key_values, 0)
    k_content = captured["k"].transpose(1, 2)  # [B, n_kv, T, d]
    codec = RopeCodec.from_model(src_model)
    assert torch.allclose(codec.apply(k_content, torch.arange(12)), k_cache, atol=1e-5)
    assert torch.allclose(codec.strip(k_cache, torch.arange(12)), k_content, atol=1e-5)
