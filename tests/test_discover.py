import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from kvtransfer.discover import (classify, enumerate_pairs, format_models, format_pairs, hf_suggestion,
                                 mismatched_kv_neighbours, scan, scan_ollama, format_ollama)


def _write_model(d: Path, cfg: dict, tokenizer_bytes: bytes = b"tok-A", weights=True):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "tokenizer.json").write_bytes(tokenizer_bytes)
    if weights:
        save_file({"w": torch.zeros(4, 4)}, str(d / "model.safetensors"))


def _cfg(model_type="qwen3", L=4, kv=2, heads=4, dh=16, hidden=64, **extra):
    return {"model_type": model_type, "architectures": ["X"], "num_hidden_layers": L, "num_key_value_heads": kv,
            "num_attention_heads": heads, "head_dim": dh, "hidden_size": hidden, "intermediate_size": 128,
            "vocab_size": 256, "rope_theta": 10000.0, "torch_dtype": "bfloat16", **extra}


def test_scan_plain_dirs_and_hub_cache_layout(tmp_path):
    _write_model(tmp_path / "models" / "a-small", _cfg(L=3))
    _write_model(tmp_path / "models" / "a-large", _cfg(L=5))
    snap = tmp_path / "hub" / "models--org--b-mid" / "snapshots" / "abc"
    _write_model(snap, _cfg(L=4, kv=4), tokenizer_bytes=b"tok-B")
    models = scan([tmp_path / "models", tmp_path / "hub"], include_hf_cache=False)
    names = sorted(m.name for m in models)
    assert names == ["a-large", "a-small", "org/b-mid"]
    a = next(m for m in models if m.name == "a-small")
    assert a.supported and a.n_layers == 3 and a.n_kv == 2 and a.head_dim == 16 and a.weight_bytes > 0
    pairs = enumerate_pairs(models)
    assert {(p.source.name, p.target.name) for p in pairs} == {("a-small", "a-large"), ("a-large", "a-small")}
    assert pairs[0].direction == "small->large"
    assert "a-small" in format_models(models) and "a-large" in format_pairs(pairs)


def test_classify_unsupported_cases(tmp_path):
    assert classify(_cfg(model_type="nemotron_h"))[0].startswith("unsupported: hybrid")
    assert classify(_cfg(model_type="deepseek_v3", kv_lora_rank=512))[0].startswith("unsupported: MLA")
    assert classify(_cfg(model_type="gemma3_text", layer_types=["sliding_attention", "full_attention"]))[0].startswith("unsupported: sliding")
    q = classify(_cfg(quantization_config={"quant_method": "awq"}))
    assert q[0].startswith("supported (quantized")
    moe = classify(_cfg(model_type="qwen3_moe", num_experts=8))
    assert moe[0] == "supported" and any("MoE" in n for n in moe[1])
    d = tmp_path / "gguf"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(_cfg()))
    (d / "x.gguf").write_bytes(b"GGUF")
    assert "GGUF" in classify(_cfg(), d)[0]


def test_mismatched_and_cross_tokenizer_are_not_paired(tmp_path):
    _write_model(tmp_path / "m" / "x2", _cfg(L=3, kv=2))
    _write_model(tmp_path / "m" / "x4", _cfg(L=3, kv=4))                      # same tokenizer, mismatched KV
    _write_model(tmp_path / "m" / "y2", _cfg(L=3, kv=2), tokenizer_bytes=b"other")  # different tokenizer
    models = scan([tmp_path / "m"], include_hf_cache=False)
    assert enumerate_pairs(models) == []
    mm = mismatched_kv_neighbours(models)
    assert len(mm) == 1 and {mm[0][0].name, mm[0][1].name} == {"x2", "x4"}


def test_ollama_manifests_and_hf_suggestions(tmp_path):
    root = tmp_path / "ollama"
    man = root / "manifests" / "registry.ollama.ai" / "library" / "qwen2.5" / "7b-instruct"
    man.parent.mkdir(parents=True)
    man.write_text(json.dumps({"layers": [{"size": 4_000_000_000}]}))
    (root / "manifests" / "registry.ollama.ai" / "library" / "gemma4").mkdir(parents=True)
    (root / "manifests" / "registry.ollama.ai" / "library" / "gemma4" / "26b").write_text(json.dumps({"layers": []}))
    found = scan_ollama([root])
    tags = {m.tag: m.hf_suggestion for m in found}
    assert tags["qwen2.5:7b-instruct"] == "Qwen/Qwen2.5-7B-Instruct"
    assert tags["gemma4:26b"] == "google/gemma-4-26b-it"
    assert "not usable directly" in format_ollama(found)
    assert hf_suggestion("qwen3:30b-a3b-instruct-2507-q4_K_M") == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert hf_suggestion("llama3.2:1b") == "meta-llama/Llama-3.2-1B-Instruct"
    assert hf_suggestion("mystery:1b") is None
