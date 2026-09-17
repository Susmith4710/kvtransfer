"""Latency: mapper application vs. target re-prefill (paper Sec. 4.7 / Appendix G).

Re-prefill runs the target's transformer body (``model.model``) without the LM head, as in the
paper.  The mapper runs in eager mode.  Inputs are synthetic; the source prefill is *not* part of
either number, because in the deployment scenario it has already happened.  The paper's protocol is
50 warmup + 30 timed trials over ten lengths 64..32768 (App. G); ``benchmark``'s own defaults are
smaller for quick checks, and ``kvtransfer experiment`` uses the paper's constants.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .hf import cache_to_list, prefill
from .mapper import Mapper
from .rope import RopeCodec


@dataclass
class BenchRow:
    seq_len: int
    mapper_ms: float
    reprefill_ms: float

    @property
    def speedup(self) -> float:
        return self.reprefill_ms / self.mapper_ms if self.mapper_ms > 0 else float("nan")


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _timeit(fn, dev, warmup: int, trials: int) -> float:
    for _ in range(warmup):
        fn()
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(trials):
        fn()
    _sync(dev)
    return (time.perf_counter() - t0) * 1000.0 / trials


@torch.no_grad()
def benchmark(source_model, target_model, mapper: Mapper, seq_lens=(64, 512, 2048), warmup: int = 5,
              trials: int = 10, batch: int = 1, progress: bool = False) -> list[BenchRow]:
    src_codec, tgt_codec = RopeCodec.from_model(source_model), RopeCodec.from_model(target_model)
    src_dev, tgt_dev = next(source_model.parameters()).device, next(target_model.parameters()).device
    tgt_dtype = next(target_model.parameters()).dtype
    vocab = int(getattr(target_model.config, "vocab_size", 1000))
    mapper.to(device=src_dev)
    base = getattr(target_model, "model", target_model)
    rows = []
    for T in seq_lens:
        ids = torch.randint(0, vocab, (batch, T))
        _, src_cache = prefill(source_model, ids.to(src_dev))
        kvs = cache_to_list(src_cache, clone=True)
        pos = torch.arange(T, device=src_dev)

        def run_mapper():
            mapped = mapper.apply_kv(kvs, pos, src_codec, tgt_codec, out_dtype=tgt_dtype)
            if tgt_dev != src_dev:  # include the cross-device shipment of the mapped cache
                mapped = [(k.to(tgt_dev), v.to(tgt_dev)) for k, v in mapped]
            return mapped

        ids_t = ids.to(tgt_dev)

        def run_reprefill():
            return base(input_ids=ids_t, use_cache=True)

        m_ms = _timeit(run_mapper, src_dev, warmup, trials)
        p_ms = _timeit(run_reprefill, tgt_dev, warmup, trials)
        rows.append(BenchRow(T, m_ms, p_ms))
        if progress:
            print(f"[bench] T={T}: mapper {m_ms:.1f} ms, re-prefill {p_ms:.1f} ms, {rows[-1].speedup:.1f}x", flush=True)
    return rows


def format_rows(rows: list[BenchRow]) -> str:
    lines = [f"{'seq_len':>8} {'mapper_ms':>10} {'reprefill_ms':>13} {'speedup':>8}"]
    for r in rows:
        lines.append(f"{r.seq_len:>8} {r.mapper_ms:>10.1f} {r.reprefill_ms:>13.1f} {r.speedup:>7.1f}x")
    return "\n".join(lines)
