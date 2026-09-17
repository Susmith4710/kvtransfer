"""Calibration pass: run source and target on the same tokens and accumulate ridge statistics.

Paper Sec. 3.1: 500 FineWeb-Edu sequences x 1,024 tokens, stride-4 token subsampling, keys
mapped in RoPE-stripped content space, values as-is.  One pass over the data produces, per cache
kind (K, V), the full cross-layer moments between *every* source layer and *every* target layer.
Layer selection (Sec. 3.2) and the mapper fit for any ``k`` (Sec. 3.1) are then pure linear
algebra on those moments, so sweeping ``k`` as the paper does costs nothing extra.

Memory: per kind, ``Gram`` is (L_s * n_kv * d_h)^2 and ``Cross`` is (L_s * n_kv * d_h) x
(L_t * n_kv * d_h) in ``stats_dtype``.  For Qwen3-0.6B->1.7B (28->28 layers, 8x128) that is
2 x 3.2 GB in fp32; for Qwen3-14B->32B (40->64) it is 6.7 GB + 10.7 GB per kind.  Use
``kinds=("K",)`` then ``("V",)`` in two passes when memory is tight, or put the accumulator on a
GPU with ``stats_device="cuda"``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import torch
from safetensors.torch import load_file, save_file

from .hf import ModelSpec, cache_layer, check_matched_kv, model_spec
from .ridge import MomentAccumulator
from .rope import RopeCodec

KINDS = ("K", "V")


@dataclass
class CalibrationStats:
    """Cross-layer moments for one source/target pair, one accumulator per cache kind."""

    source: ModelSpec
    target: ModelSpec
    stride: int
    seq_len: int
    n_seqs: int = 0
    acc: dict = field(default_factory=dict)  # kind -> MomentAccumulator

    # ---- geometry helpers ----------------------------------------------------------------------
    def src_rows(self, layers: Sequence[int]) -> torch.Tensor:
        """Row indices of the concatenated features of the given source layers (all heads)."""
        w = self.source.kv_width
        return torch.cat([torch.arange(l * w, (l + 1) * w) for l in layers])

    def tgt_cols(self, layer: int, head: int | None = None) -> torch.Tensor:
        w, d = self.target.kv_width, self.target.head_dim
        if head is None:
            return torch.arange(layer * w, (layer + 1) * w)
        return torch.arange(layer * w + head * d, layer * w + (head + 1) * d)

    def src_rows_head(self, layer: int, head: int) -> torch.Tensor:
        w, d = self.source.kv_width, self.source.head_dim
        return torch.arange(layer * w + head * d, layer * w + (head + 1) * d)

    # ---- (de)serialisation ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "stride": self.stride,
            "seq_len": self.seq_len,
            "n_seqs": self.n_seqs,
            "kinds": list(self.acc),
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2))
        for kind, acc in self.acc.items():
            save_file(acc.state_dict(), str(path / f"stats_{kind}.safetensors"))

    @classmethod
    def load(cls, path: str | Path, device=None, kinds: Iterable[str] | None = None) -> "CalibrationStats":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        st = cls(ModelSpec(**meta["source"]), ModelSpec(**meta["target"]), meta["stride"], meta["seq_len"],
                 meta["n_seqs"])
        for kind in (kinds or meta["kinds"]):
            st.acc[kind] = MomentAccumulator.from_state_dict(
                load_file(str(path / f"stats_{kind}.safetensors")), device=device)
        return st


@torch.no_grad()
def extract_content_kv(model, codec: RopeCodec, cache, n_layers: int, positions: torch.Tensor, kind: str,
                       stride: int = 1, offset: int = 0) -> torch.Tensor:
    """Per-token features for one kind across all layers: [n_tok, n_layers * n_kv * d_h].

    Keys are RoPE-stripped (content space).  Tokens are subsampled with ``positions[offset::stride]``.
    """
    feats = []
    sel = torch.arange(offset, positions.numel(), stride, device=positions.device)
    for l in range(n_layers):
        k, v = cache_layer(cache, l)
        t = k if kind == "K" else v
        t = t[:, :, sel]  # [B, n_kv, n, d_h]
        if kind == "K":
            t = codec.strip(t.float(), positions[sel])
        B, n_kv, n, d = t.shape
        feats.append(t.permute(0, 2, 1, 3).reshape(B * n, n_kv * d).float())
    return torch.cat(feats, dim=1)


@torch.no_grad()
def calibrate(
    source_model,
    target_model,
    token_batches: Iterable[torch.Tensor],
    *,
    stride: int = 4,
    kinds: Sequence[str] = KINDS,
    stats_device: str | torch.device | None = None,
    stats_dtype: torch.dtype = torch.float32,
    source_name: str | None = None,
    target_name: str | None = None,
    require_matched_kv: bool = True,
    progress: bool = False,
) -> CalibrationStats:
    """Stream ``token_batches`` (each [B, T] LongTensor) through both models and accumulate moments."""
    src_spec = model_spec(source_model, source_name)
    tgt_spec = model_spec(target_model, target_name)
    if require_matched_kv:
        check_matched_kv(src_spec, tgt_spec)
    src_codec = RopeCodec.from_model(source_model)
    tgt_codec = RopeCodec.from_model(target_model)
    src_dev = next(source_model.parameters()).device
    tgt_dev = next(target_model.parameters()).device
    if stats_device is None:
        stats_device = src_dev

    p = src_spec.n_layers * src_spec.kv_width
    q = tgt_spec.n_layers * tgt_spec.kv_width
    stats = CalibrationStats(src_spec, tgt_spec, stride, seq_len=0)
    for kind in kinds:
        stats.acc[kind] = MomentAccumulator(p, q, device=stats_device, dtype=stats_dtype)

    for i, ids in enumerate(token_batches):
        ids = ids.long()
        B, T = ids.shape
        stats.seq_len = max(stats.seq_len, T)
        stats.n_seqs += B
        pos = torch.arange(T)
        src_out = source_model(input_ids=ids.to(src_dev), use_cache=True)
        tgt_out = target_model(input_ids=ids.to(tgt_dev), use_cache=True)
        for kind in kinds:
            x = extract_content_kv(source_model, src_codec, src_out.past_key_values, src_spec.n_layers,
                                   pos.to(src_dev), kind, stride)
            y = extract_content_kv(target_model, tgt_codec, tgt_out.past_key_values, tgt_spec.n_layers,
                                   pos.to(tgt_dev), kind, stride)
            stats.acc[kind].update(x, y)
        del src_out, tgt_out
        if progress:
            print(f"[calibrate] batch {i + 1}: {stats.n_seqs} sequences, {stats.acc[kinds[0]].n} tokens", flush=True)
    return stats
