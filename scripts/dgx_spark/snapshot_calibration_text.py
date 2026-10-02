#!/usr/bin/env python
"""Save the head of the FineWeb-Edu stream to a local .jsonl, so calibration needs no network.

Streaming the dataset inside a long run is fragile: the `datasets` streaming threads hang or crash
the interpreter at exit, and a network blip mid-calibration costs the run.  This writes the same
documents the stream yields, in the same order, until enough long ones have been seen, and
`kvtransfer ... --data <out.jsonl>` then reads them offline.

Usage:
    snapshot_calibration_text.py runs/data/fineweb-edu-head.jsonl \
        [--tokenizer Qwen/Qwen3-4B-Instruct-2507] [--n-long 800] [--min-tokens 1024]

`--n-long` documents of at least `--min-tokens` tokens are collected (the default covers the 500
calibration + 32 held-out sequences of the experiment with room to spare).  Every document seen on
the way is written, short ones included, so the file is exactly a prefix of the stream.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507", help="HF id or local path; only used to count tokens")
    ap.add_argument("--n-long", type=int, default=800)
    ap.add_argument("--min-tokens", type=int, default=1024)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--config", default="default")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = load_dataset(args.dataset, args.config, split="train", streaming=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    t0, n, n_long = time.time(), 0, 0
    with open(tmp, "w") as f:
        for ex in ds:
            text = ex["text"]
            f.write(json.dumps({"text": text}) + "\n")
            n += 1
            if len(tok(text, add_special_tokens=False)["input_ids"]) >= args.min_tokens:
                n_long += 1
                if n_long >= args.n_long:
                    break
    os.replace(tmp, args.out)
    print(f"wrote {n} documents ({n_long} with >= {args.min_tokens} tokens) to {args.out}, "
          f"{os.path.getsize(args.out) / 1e6:.1f} MB, in {time.time() - t0:.0f}s", flush=True)
    sys.stdout.flush()
    os._exit(0)   # skip interpreter finalisation: the streaming threads hang or abort there


if __name__ == "__main__":
    main()
