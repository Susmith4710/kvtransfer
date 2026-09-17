"""Transfer-quality diagnostics on held-out text (paper Sec. 4.5).

* ``cache_r2``: held-out reconstruction R^2 per target layer for K (content space) and V.
* ``attention_output_cosine``: cosine between the attention output the target computes from the
  *mapped* prefix cache and from its *own* prefix cache, on the same suffix tokens.  The paper finds
  this predicts downstream retention (r=+0.57) where R^2 does not (r=-0.20).
* ``logit_divergence``: KL(p_own || p_mapped) and top-1 agreement on the suffix tokens.

All three share the same protocol: the prompt is split into a prefix (mapped) and a suffix that
the target processes against either cache.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .hf import cache_layer, cache_num_layers, cache_to_list, decoder_layers, forward_with_cache, prefill, crop_cache
from .mapper import Mapper
from .ridge import r2_score
from .rope import RopeCodec


@dataclass
class EvalReport:
    n_sequences: int = 0
    prefix_len: int = 0
    suffix_len: int = 0
    r2_K: list = field(default_factory=list)          # per target layer, averaged over sequences
    r2_V: list = field(default_factory=list)
    attn_cosine_layers: list = field(default_factory=list)  # per target layer
    attn_cosine_mean: float = float("nan")
    attn_cosine_min: float = float("nan")
    kl_mean: float = float("nan")                      # KL(own || mapped) per suffix token, mean
    kl_p95: float = float("nan")
    top1_agreement: float = float("nan")
    source_top1_agreement: float = float("nan")        # how often the *source's* own next token matches the target's

    def summary(self) -> str:
        return (
            f"{self.n_sequences} seqs, prefix {self.prefix_len}, suffix {self.suffix_len}\n"
            f"  held-out R2   K={np.nanmean(self.r2_K):.3f}  V={np.nanmean(self.r2_V):.3f}\n"
            f"  attn cosine   mean={self.attn_cosine_mean:.3f}  min-layer={self.attn_cosine_min:.3f}\n"
            f"  logit KL      mean={self.kl_mean:.3f}  p95={self.kl_p95:.3f}\n"
            f"  top-1 agree   mapped-vs-own={self.top1_agreement:.3f}  source-vs-target={self.source_top1_agreement:.3f}"
        )

    def to_dict(self) -> dict:
        return {k: (v if not isinstance(v, float) or np.isfinite(v) else None) for k, v in self.__dict__.items()}


class _AttnOutputTap:
    """Captures the attention output (input of o_proj) of every decoder layer of a model."""

    def __init__(self, model):
        self.outs = {}
        self.handles = []
        for i, layer in enumerate(decoder_layers(model)):
            proj = layer.self_attn.o_proj
            self.handles.append(proj.register_forward_pre_hook(self._hook(i)))

    def _hook(self, i):
        def fn(module, args):
            self.outs[i] = args[0].detach().float()
        return fn

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def evaluate(source_model, target_model, mapper: Mapper, sequences, prefix_len: int, suffix_len: int = 32,
             progress: bool = False) -> EvalReport:
    """Run all diagnostics over ``sequences`` (iterable of [1, T] LongTensors with T >= prefix+suffix)."""
    src_codec, tgt_codec = RopeCodec.from_model(source_model), RopeCodec.from_model(target_model)
    src_dev, tgt_dev = next(source_model.parameters()).device, next(target_model.parameters()).device
    mapper.to(device=src_dev)
    Lt = mapper.target.n_layers
    rep = EvalReport(prefix_len=prefix_len, suffix_len=suffix_len)
    r2k = np.zeros(Lt)
    r2v = np.zeros(Lt)
    cos = np.zeros(Lt)
    kls, agree, s_agree = [], [], []
    pos = torch.arange(prefix_len)

    for ids in sequences:
        ids = ids.long()
        if ids.dim() == 1:
            ids = ids[None]
        if ids.shape[1] < prefix_len + suffix_len:
            continue
        prefix = ids[:, :prefix_len]
        suffix = ids[:, prefix_len:prefix_len + suffix_len]
        rep.n_sequences += 1

        src_logits, src_cache = prefill(source_model, prefix.to(src_dev))
        own_logits, own_cache = prefill(target_model, prefix.to(tgt_dev))
        kvs = cache_to_list(src_cache, clone=False)
        mapped = mapper.apply_kv(kvs, pos.to(src_dev), src_codec, tgt_codec, out_dtype=torch.float32)

        # held-out reconstruction R^2 in the spaces the mapper predicts (content-space K, raw V)
        for l in range(Lt):
            k_own, v_own = cache_layer(own_cache, l)
            k_hat, v_hat = mapped[l]
            k_own_c = tgt_codec.strip(k_own.float(), pos.to(tgt_dev)).reshape(-1, mapper.target.head_dim)
            k_hat_c = tgt_codec.strip(k_hat.float().to(tgt_dev), pos.to(tgt_dev)).reshape(-1, mapper.target.head_dim)
            r2k[l] += r2_score(k_own_c, k_hat_c)
            r2v[l] += r2_score(v_own.float().reshape(-1, mapper.target.head_dim),
                               v_hat.float().to(tgt_dev).reshape(-1, mapper.target.head_dim))

        # attention-output cosine + logits on the suffix under own vs mapped prefix cache
        from .hf import list_to_cache
        tgt_dtype = next(target_model.parameters()).dtype
        mapped_cache = list_to_cache([(k.to(tgt_dev, tgt_dtype), v.to(tgt_dev, tgt_dtype)) for k, v in mapped], target_model)
        tap = _AttnOutputTap(target_model)
        out_own = forward_with_cache(target_model, own_cache, suffix.to(tgt_dev), past_len=prefix_len)
        own_attn = dict(tap.outs)
        tap.outs.clear()
        out_map = forward_with_cache(target_model, mapped_cache, suffix.to(tgt_dev), past_len=prefix_len)
        map_attn = dict(tap.outs)
        tap.close()
        for l in range(Lt):
            a, b = own_attn[l].reshape(-1, own_attn[l].shape[-1]), map_attn[l].reshape(-1, map_attn[l].shape[-1])
            cos[l] += float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())
        lp_own = torch.log_softmax(out_own.logits.float(), -1)
        lp_map = torch.log_softmax(out_map.logits.float(), -1)
        kl = (lp_own.exp() * (lp_own - lp_map)).sum(-1).reshape(-1)
        kls.append(kl.cpu())
        agree.append((lp_own.argmax(-1) == lp_map.argmax(-1)).float().reshape(-1).cpu())
        # source's own prediction at the last prefix position vs target's own
        s_agree.append(float(src_logits[:, -1].argmax(-1).cpu() == own_logits[:, -1].argmax(-1).cpu()))
        if progress:
            print(f"[eval] seq {rep.n_sequences}: KL={float(kl.mean()):.3f} agree={float(agree[-1].mean()):.2f}", flush=True)

    n = max(rep.n_sequences, 1)
    rep.r2_K = (r2k / n).tolist()
    rep.r2_V = (r2v / n).tolist()
    rep.attn_cosine_layers = (cos / n).tolist()
    if rep.n_sequences:
        rep.attn_cosine_mean = float(np.mean(rep.attn_cosine_layers))
        rep.attn_cosine_min = float(np.min(rep.attn_cosine_layers))
        allkl = torch.cat(kls)
        rep.kl_mean = float(allkl.mean())
        rep.kl_p95 = float(torch.quantile(allkl, 0.95))
        rep.top1_agreement = float(torch.cat(agree).mean())
        rep.source_top1_agreement = float(np.mean(s_agree))
    return rep


@torch.no_grad()
def cache_r2(mapper: Mapper, source_model, target_model, ids: torch.Tensor) -> dict:
    """Quick per-layer held-out R^2 for a single sequence (subset of :func:`evaluate`)."""
    rep = evaluate(source_model, target_model, mapper, [ids], prefix_len=ids.shape[-1] - 1, suffix_len=1)
    return {"K": rep.r2_K, "V": rep.r2_V}
