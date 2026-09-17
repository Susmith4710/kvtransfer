"""Calibration / evaluation text sources.

The paper calibrates on FineWeb-Edu (500 x 1,024 tokens) and finds Wikipedia within ~1 pp and code
~5 pp worse (Appendix C), so use general web/educational prose unless your workload is narrow.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import torch


def iter_text_file(path: str | Path) -> Iterator[str]:
    """Yield documents from a ``.txt`` (blank-line separated), ``.jsonl`` (``text`` field) or ``.json`` list."""
    path = Path(path)
    if path.suffix == ".jsonl":
        import json
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    yield d["text"] if isinstance(d, dict) else str(d)
    elif path.suffix == ".json":
        import json
        for d in json.loads(path.read_text()):
            yield d["text"] if isinstance(d, dict) else str(d)
    else:
        buf = []
        with path.open() as f:
            for line in f:
                if line.strip():
                    buf.append(line.rstrip("\n"))
                elif buf:
                    yield "\n".join(buf)
                    buf = []
        if buf:
            yield "\n".join(buf)


def iter_fineweb_edu(split: str = "train", name: str = "default") -> Iterator[str]:
    """Stream FineWeb-Edu documents (requires the ``datasets`` package and Hub access)."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name, split=split, streaming=True)
    for ex in ds:
        yield ex["text"]


def iter_dataset(spec: str) -> Iterator[str]:
    """``fineweb-edu`` | ``hf:<dataset>[:config][:split]`` (needs a ``text`` column) | path to a local file."""
    if spec == "fineweb-edu":
        return iter_fineweb_edu()
    if spec.startswith("hf:"):
        from datasets import load_dataset
        parts = spec[3:].split(":")
        name, cfg, split = parts[0], (parts[1] if len(parts) > 1 else None), (parts[2] if len(parts) > 2 else "train")
        ds = load_dataset(name, cfg, split=split, streaming=True)
        return (ex["text"] for ex in ds)
    return iter_text_file(spec)


def token_sequences(texts: Iterable[str], tokenizer, seq_len: int, n_seqs: int, min_len: int | None = None,
                    add_special_tokens: bool = False) -> list[torch.Tensor]:
    """Tokenize documents and keep the first ``seq_len`` tokens of those at least ``min_len`` long."""
    min_len = min_len or seq_len
    out = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]
        if len(ids) >= min_len:
            out.append(torch.tensor(ids[:seq_len], dtype=torch.long))
        if len(out) >= n_seqs:
            break
    if len(out) < n_seqs:
        raise ValueError(f"only {len(out)} of {n_seqs} requested sequences of >= {min_len} tokens were found")
    return out


def batches(seqs: list[torch.Tensor], batch_size: int) -> Iterator[torch.Tensor]:
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        T = min(s.numel() for s in chunk)
        yield torch.stack([s[:T] for s in chunk])
