"""Runtime: source prefill -> mapped cache -> target decode, on Hugging Face transformers models.

This is the pipeline of paper Fig. 1.  The paper's re-prefill baseline runs the receiver's
transformer body; here the receiver instead starts from the mapped cache.  One protocol detail the
paper leaves implicit: to produce the *first* target logit you need the target's own hidden state
somewhere, so the last ``hold_back`` prompt tokens are withheld from the mapping and run through
the target against the mapped prefix (the ``generate`` adapter does this for you).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .hf import cache_to_list, forward_with_cache, list_to_cache, model_spec, prefill
from .mapper import Mapper
from .rope import RopeCodec


@dataclass
class TransferResult:
    tokens: torch.Tensor           # [B, T_prompt + n_new]
    n_prompt: int
    n_mapped: int                  # prompt positions covered by the mapped cache
    target_cache: object           # the target's DynamicCache after generation


class CrossModelTransfer:
    """Holds a (source, target, mapper) triple and performs handoffs."""

    def __init__(self, source_model, target_model, mapper: Mapper, check: bool = True):
        self.source = source_model
        self.target = target_model
        self.mapper = mapper
        self.src_codec = RopeCodec.from_model(source_model)
        self.tgt_codec = RopeCodec.from_model(target_model)
        self.src_dev = next(source_model.parameters()).device
        self.tgt_dev = next(target_model.parameters()).device
        self.tgt_dtype = next(target_model.parameters()).dtype
        if check:
            s, t = model_spec(source_model), model_spec(target_model)
            for got, want, who in ((s, mapper.source, "source"), (t, mapper.target, "target")):
                if (got.n_layers, got.n_kv, got.head_dim) != (want.n_layers, want.n_kv, want.head_dim):
                    raise ValueError(f"{who} model geometry {got} does not match the mapper's {want}")
        dev_w = self.src_dev
        self.mapper.to(device=dev_w)

    # ---- primitives -------------------------------------------------------------------------
    @torch.no_grad()
    def source_prefill(self, input_ids: torch.Tensor):
        """Run the prompt through the source; returns ``(logits, source_cache)``."""
        return prefill(self.source, input_ids.to(self.src_dev))

    @torch.no_grad()
    def map_cache(self, source_cache, n_tokens: int | None = None):
        """Map the first ``n_tokens`` positions of a source cache into a target cache (on the target device)."""
        kvs = cache_to_list(source_cache, clone=False)
        if n_tokens is not None:
            kvs = [(k[:, :, :n_tokens], v[:, :, :n_tokens]) for k, v in kvs]
        T = kvs[0][0].shape[2]
        positions = torch.arange(T, device=kvs[0][0].device)
        mapped = self.mapper.apply_kv(kvs, positions, self.src_codec, self.tgt_codec, out_dtype=self.tgt_dtype)
        mapped = [(k.to(self.tgt_dev), v.to(self.tgt_dev)) for k, v in mapped]
        return list_to_cache(mapped, self.target)

    @torch.no_grad()
    def handoff(self, input_ids: torch.Tensor, hold_back: int = 1):
        """Source prefill + mapping of all but the last ``hold_back`` prompt tokens, then the target
        consumes those tokens.  Returns ``(target_logits_for_held_tokens, target_cache)``."""
        input_ids = input_ids.to(self.src_dev)
        T = input_ids.shape[1]
        if not 1 <= hold_back <= T:
            raise ValueError("hold_back must be in [1, prompt length]")
        _, src_cache = self.source_prefill(input_ids)
        n_mapped = T - hold_back
        tgt_cache = self.map_cache(src_cache, n_mapped)
        tail = input_ids[:, n_mapped:].to(self.tgt_dev)
        out = forward_with_cache(self.target, tgt_cache, tail, past_len=n_mapped)
        return out.logits, out.past_key_values

    # ---- generation ---------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64, hold_back: int = 1,
                 eos_token_id: int | None = None, do_sample: bool = False, temperature: float = 1.0) -> TransferResult:
        """Greedy (or sampled) decoding on the target from a cache produced by the source."""
        input_ids = input_ids.to(self.src_dev)
        B, T = input_ids.shape
        logits, cache = self.handoff(input_ids, hold_back)
        seq = input_ids.to(self.tgt_dev)
        finished = torch.zeros(B, dtype=torch.bool, device=self.tgt_dev)
        for _ in range(max_new_tokens):
            nxt = _pick(logits[:, -1], do_sample, temperature)
            if eos_token_id is not None:
                nxt = torch.where(finished, torch.full_like(nxt, eos_token_id), nxt)
                finished |= nxt == eos_token_id
            seq = torch.cat([seq, nxt[:, None]], dim=1)
            # always run the chosen token through the model, EOS included, so the cache covers every
            # token in `seq` (a shorter cache would silently misalign later positions)
            out = forward_with_cache(self.target, cache, nxt[:, None], past_len=seq.shape[1] - 1)
            logits, cache = out.logits, out.past_key_values
            if eos_token_id is not None and bool(finished.all()):
                break
        return TransferResult(seq, T, T - hold_back, cache)


def _pick(logits: torch.Tensor, do_sample: bool, temperature: float) -> torch.Tensor:
    if not do_sample:
        return logits.argmax(-1)
    probs = torch.softmax(logits.float() / max(temperature, 1e-5), dim=-1)
    return torch.multinomial(probs, 1)[:, 0]


class Session:
    """Multi-turn handoff (paper Sec. 4.6): a conversation whose cache is carried across models.

    ``mappers`` maps ``(from_name, to_name)`` to a :class:`Mapper`.  The session keeps a single
    "live" model and cache; ``switch_to`` maps the whole current cache into the other model and
    re-runs the last ``hold_back`` tokens there so the new model has its own logits to continue from.
    """

    def __init__(self, models: dict, mappers: dict, start: str):
        self.models = models
        self.mappers = mappers
        self.live = start
        self.tokens = None
        self.cache = None
        self.last_logits = None

    @property
    def model(self):
        return self.models[self.live]

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def feed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Append tokens to the conversation on the live model; returns the logits of the fed tokens."""
        ids = input_ids.to(self.device)
        if self.cache is None:
            out = self.model(input_ids=ids, use_cache=True)
            self.tokens = ids
        else:
            out = forward_with_cache(self.model, self.cache, ids, past_len=self.tokens.shape[1])
            self.tokens = torch.cat([self.tokens, ids], dim=1)
        self.cache = out.past_key_values
        self.last_logits = out.logits[:, -1]
        return out.logits

    @torch.no_grad()
    def switch_to(self, name: str, hold_back: int = 1):
        """Map the current cache into model ``name``; the last ``hold_back`` tokens are recomputed there."""
        if name == self.live:
            return self.last_logits
        mapper = self.mappers[(self.live, name)]
        xfer = CrossModelTransfer(self.models[self.live], self.models[name], mapper, check=False)
        T = self.tokens.shape[1]
        n_mapped = T - hold_back
        tgt_cache = xfer.map_cache(self.cache, n_mapped)
        tail = self.tokens[:, n_mapped:].to(xfer.tgt_dev)
        out = forward_with_cache(self.models[name], tgt_cache, tail, past_len=n_mapped)
        self.cache = out.past_key_values
        self.tokens = self.tokens.to(xfer.tgt_dev)
        self.last_logits = out.logits[:, -1]
        self.live = name
        return out.logits

    @torch.no_grad()
    def generate(self, max_new_tokens: int, eos_token_id: int | None = None) -> torch.Tensor:
        """Greedy-decode on the live model, extending the conversation; returns the new tokens [B, n]."""
        if self.last_logits is None:
            raise RuntimeError("feed() a prompt first")
        logits = self.last_logits
        new = []
        for _ in range(max_new_tokens):
            nxt = logits.argmax(-1)
            new.append(nxt)
            self.tokens = torch.cat([self.tokens, nxt[:, None].to(self.device)], dim=1)
            out = forward_with_cache(self.model, self.cache, nxt[:, None], past_len=self.tokens.shape[1] - 1)
            logits, self.cache = out.logits[:, -1], out.past_key_values   # cache now covers all tokens
            if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                break
        self.last_logits = logits
        return torch.stack(new, dim=1)
