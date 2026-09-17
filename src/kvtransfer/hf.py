"""Hugging Face transformers glue: model introspection, cache access, forward-with-injected-cache.

Works with transformers >= 4.56 (layered ``DynamicCache``) and the 5.x line (``rope_parameters``
dict, ``dtype=`` kwarg).  Only dense full-attention decoder models with a model-level rotary
embedding (Llama, Qwen2/3, Mistral/Ministral, Gemma-style) are supported, which is exactly the
regime the paper evaluates.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .rope import RopeCodec


@dataclass(frozen=True)
class ModelSpec:
    """The architectural axes the mapper depends on (paper Table 10)."""

    name: str
    n_layers: int
    n_kv: int
    head_dim: int
    rope_theta: float
    attention_scaling: float

    @property
    def kv_width(self) -> int:
        """All KV heads of one layer concatenated: n_kv * head_dim."""
        return self.n_kv * self.head_dim

    def to_dict(self) -> dict:
        return asdict(self)


def _rope_theta(config) -> float:
    rp = getattr(config, "rope_parameters", None)
    if isinstance(rp, dict) and "rope_theta" in rp:
        return float(rp["rope_theta"])
    if getattr(config, "rope_theta", None) is not None:
        return float(config.rope_theta)
    raise ValueError("cannot determine rope_theta from config")


def model_spec(model: torch.nn.Module, name: str | None = None) -> ModelSpec:
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", None) or cfg  # multimodal wrappers (e.g. Ministral 3)
    head_dim = getattr(text_cfg, "head_dim", None) or text_cfg.hidden_size // text_cfg.num_attention_heads
    codec = RopeCodec.from_model(model)
    return ModelSpec(
        name=name or getattr(cfg, "_name_or_path", "") or type(model).__name__,
        n_layers=int(text_cfg.num_hidden_layers),
        n_kv=int(text_cfg.num_key_value_heads),
        head_dim=int(head_dim),
        rope_theta=_rope_theta(text_cfg),
        attention_scaling=codec.attention_scaling,
    )


def check_matched_kv(src: ModelSpec, tgt: ModelSpec) -> None:
    """Paper Sec. 2.1: matched-KV means same KV head count and per-head dim; depth may differ."""
    problems = []
    if src.n_kv != tgt.n_kv:
        problems.append(f"kv heads {src.n_kv} != {tgt.n_kv}")
    if src.head_dim != tgt.head_dim:
        problems.append(f"head dim {src.head_dim} != {tgt.head_dim}")
    if problems:
        raise ValueError(
            "not a matched-KV pair (" + "; ".join(problems) + "). The paper only validates matched-KV "
            "pairs; the ridge solve would still run but nothing is known about transfer quality."
        )


def tokenizer_fingerprint(tok) -> str:
    vocab = tok.get_vocab()
    h = hashlib.sha256()
    for k in sorted(vocab, key=lambda s: vocab[s]):
        h.update(f"{vocab[k]}:{k}\n".encode())
    return h.hexdigest()[:16]


SAMPLE_TEXTS = (
    "The quick brown fox jumps over the lazy dog. 1234567890 !@#$%^&*()",
    "def f(x):\n    return {'a': x ** 2, 'b': [x, x + 1]}\n",
    "Photosynthesis converts light energy into chemical energy in plants, algae and some bacteria.",
    "Résumé — naïve café; 東京 · Москва · القاهرة · 🙂",
)


def tokenizer_compatibility(tok_a, tok_b, texts=SAMPLE_TEXTS) -> str:
    """'identical' (same vocab map), 'compatible' (same ids on sample texts, e.g. Qwen2.5 vs Qwen3
    tokenizers, which share the BPE), or 'incompatible' (token positions would not align)."""
    if tok_a.get_vocab() == tok_b.get_vocab():
        return "identical"
    for t in texts:
        if tok_a(t, add_special_tokens=False)["input_ids"] != tok_b(t, add_special_tokens=False)["input_ids"]:
            return "incompatible"
    return "compatible"


def assert_shared_tokenizer(tok_a, tok_b, allow_compatible: bool = True) -> str:
    level = tokenizer_compatibility(tok_a, tok_b)
    if level == "incompatible" or (level == "compatible" and not allow_compatible):
        raise ValueError(f"source and target tokenizers are {level}; token positions would not align")
    return level


def load_model(model_id: str, device: str | torch.device | None = None, dtype=None, attn_implementation=None,
               device_map=None, **kwargs) -> torch.nn.Module:
    """Load a causal LM in eval mode.  bf16 on CUDA, fp32 on CPU unless ``dtype`` is given.

    ``attn_implementation`` defaults to ``"sdpa"``: it is what works everywhere, including the DGX
    Spark (sm_121), where flash-attn has no wheels.  Pass ``"flash_attention_2"`` explicitly to use it.
    ``device_map`` (e.g. ``"auto"``) is forwarded to transformers for sharded loading; the model is
    then left where transformers put it instead of being moved to ``device``.
    """
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    kw = dict(kwargs)
    kw["attn_implementation"] = attn_implementation or "sdpa"
    if device_map is not None:
        kw["device_map"] = device_map
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **kw)
    except TypeError:  # transformers < 5
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **kw)
    if device_map is None:
        model = model.to(dev)
    return model.eval()


def load_pair(source_id: str, target_id: str, device=None, dtype=None, attn_implementation=None, **kwargs):
    """Load source and target (same device/dtype) plus the shared tokenizer, checking the tokenizers agree."""
    src = load_model(source_id, device=device, dtype=dtype, attn_implementation=attn_implementation, **kwargs)
    tgt = load_model(target_id, device=device, dtype=dtype, attn_implementation=attn_implementation, **kwargs)
    tok = load_tokenizer(source_id, **{k: v for k, v in kwargs.items() if k == "trust_remote_code"})
    assert_shared_tokenizer(tok, load_tokenizer(target_id, **{k: v for k, v in kwargs.items() if k == "trust_remote_code"}))
    return src, tgt, tok


def encode_prompt(tokenizer, prompt: str | list, chat: bool = False, system: str | None = None,
                  enable_thinking: bool | None = None) -> torch.Tensor:
    """Tokenize a raw string or, with ``chat=True``, a user message (or a list of chat messages)
    through the tokenizer's chat template with the generation prompt appended.  Returns [1, T]."""
    if not chat:
        return tokenizer(prompt, return_tensors="pt")["input_ids"]
    msgs = prompt if isinstance(prompt, list) else ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}]
    kw = {}
    if enable_thinking is not None:
        kw["enable_thinking"] = enable_thinking   # Qwen3 templates accept this
    try:
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", **kw)
    except TypeError:
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    return ids


def load_tokenizer(model_id: str, **kwargs):
    return AutoTokenizer.from_pretrained(model_id, **kwargs)


# --------------------------------------------------------------------------------------- caches

def cache_layer(cache, layer_idx: int):
    """(K, V) of one layer, each [B, n_kv, T, d_h], for either cache layout."""
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values
    return cache.key_cache[layer_idx], cache.value_cache[layer_idx]


def cache_num_layers(cache) -> int:
    if hasattr(cache, "layers"):
        return len(cache.layers)
    return len(cache.key_cache)


def cache_to_list(cache, n_layers: int | None = None, clone: bool = True):
    n = n_layers if n_layers is not None else cache_num_layers(cache)
    out = []
    for l in range(n):
        k, v = cache_layer(cache, l)
        out.append((k.detach().clone(), v.detach().clone()) if clone else (k, v))
    return out


def list_to_cache(kvs, model=None) -> DynamicCache:
    """Build a ``DynamicCache`` from a list over layers of ``(K, V)`` with shape [B, n_kv, T, d_h]."""
    cache = _new_cache(model)
    for l, (k, v) in enumerate(kvs):
        cache.update(k.contiguous(), v.contiguous(), l)
    return cache


def _new_cache(model=None) -> DynamicCache:
    if model is not None:
        try:
            return DynamicCache(config=model.config)
        except TypeError:
            pass
    return DynamicCache()


def crop_cache(cache, n_tokens: int):
    if hasattr(cache, "crop"):
        cache.crop(n_tokens)
        return cache
    kvs = [(k[:, :, :n_tokens], v[:, :, :n_tokens]) for k, v in cache_to_list(cache)]
    return list_to_cache(kvs)


@torch.no_grad()
def prefill(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
    """Run the prompt and return ``(logits, cache)``."""
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    return out.logits, out.past_key_values


@torch.no_grad()
def forward_with_cache(model, cache, input_ids: torch.Tensor, past_len: int | None = None):
    """Run ``input_ids`` [B, C] at positions ``past_len .. past_len+C-1`` on top of an injected cache."""
    B, C = input_ids.shape
    if past_len is None:
        past_len = cache.get_seq_length()
    dev = input_ids.device
    attn = torch.ones(B, past_len + C, dtype=torch.long, device=dev)
    pos = torch.arange(past_len, past_len + C, device=dev)
    return model(
        input_ids=input_ids,
        past_key_values=cache,
        attention_mask=attn,
        position_ids=pos[None].expand(B, -1),
        cache_position=pos,
        use_cache=True,
    )


def decoder_layers(model) -> list:
    for path in (("model", "layers"), ("model", "language_model", "layers"), ("language_model", "model", "layers"),
                 ("language_model", "layers"), ("layers",)):
        obj = model
        for name in path:
            obj = getattr(obj, name, None)
            if obj is None:
                break
        else:
            return list(obj)
    raise ValueError("cannot locate decoder layers on this model")


def transformer_body(model):
    """The decoder without the LM head (what the paper times as 're-prefill')."""
    base = getattr(model, "model", model)
    return base


def spec_json(spec: ModelSpec) -> str:
    return json.dumps(spec.to_dict(), indent=2)
