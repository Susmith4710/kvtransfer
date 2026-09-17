"""Tiny random-init models so the suite runs offline on CPU in seconds."""
from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

VOCAB = 257


def tiny_qwen3(n_layers: int, seed: int, n_kv: int = 2, head_dim: int = 16, hidden: int = 64, rope_scaling=None):
    torch.manual_seed(seed)
    kw = dict(
        vocab_size=VOCAB, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=n_layers,
        num_attention_heads=4, num_key_value_heads=n_kv, head_dim=head_dim, max_position_embeddings=512,
        rope_theta=10000.0, tie_word_embeddings=False, attn_implementation="eager",
    )
    if rope_scaling:
        kw["rope_scaling"] = rope_scaling
    cfg = Qwen3Config(**kw)
    return Qwen3ForCausalLM(cfg).eval()


def tiny_llama(n_layers: int, seed: int, n_kv: int = 2, head_dim: int = 16, hidden: int = 64):
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=n_layers,
        num_attention_heads=hidden // head_dim, num_key_value_heads=n_kv, head_dim=head_dim,
        max_position_embeddings=512, rope_theta=500000.0, tie_word_embeddings=False, attn_implementation="eager",
    )
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture(scope="session")
def src_model():
    return tiny_qwen3(3, seed=1)


@pytest.fixture(scope="session")
def tgt_model():
    return tiny_qwen3(4, seed=2)


@pytest.fixture(scope="session")
def llama_model():
    return tiny_llama(3, seed=3)


def random_batches(n_batches: int, batch: int, T: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, VOCAB, (batch, T), generator=g) for _ in range(n_batches)]


@pytest.fixture(scope="session")
def calib_batches():
    return random_batches(n_batches=6, batch=4, T=48, seed=0)
