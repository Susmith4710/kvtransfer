"""Command line: check a pair, fit mappers, evaluate, benchmark, generate.

    kvtransfer check    --source Qwen/Qwen3-0.6B --target Qwen/Qwen3-1.7B --k 8
    kvtransfer fit      --source Qwen/Qwen3-0.6B --target Qwen/Qwen3-1.7B --data fineweb-edu \\
                        --n-seqs 500 --seq-len 1024 --k 1,4,8 --out mappers/qwen3-0.6b-to-1.7b
    kvtransfer eval     --source ... --target ... --mapper mappers/qwen3-0.6b-to-1.7b/k8 --data fineweb-edu
    kvtransfer bench    --source ... --target ... --mapper ... --seq-lens 64,512,2048,8192
    kvtransfer generate --source ... --target ... --mapper ... --prompt "..."
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _add_pair(p: argparse.ArgumentParser):
    p.add_argument("--source", required=True, help="HF model id or local path of the model that prefills")
    p.add_argument("--target", required=True, help="HF model id or local path of the model that decodes")
    p.add_argument("--device", default=None, help="cuda | cpu | cuda:0 (default: cuda if available)")
    p.add_argument("--source-device", default=None, help="override device for the source model")
    p.add_argument("--target-device", default=None, help="override device for the target model")
    p.add_argument("--dtype", default=None, help="bfloat16 | float16 | float32 (default: bf16 on cuda, fp32 on cpu)")
    p.add_argument("--attn", default=None, help="attn_implementation, e.g. flash_attention_2 or sdpa")
    p.add_argument("--trust-remote-code", action="store_true")


def _add_data(p: argparse.ArgumentParser, n_seqs: int, seq_len: int):
    p.add_argument("--data", default="fineweb-edu", help="fineweb-edu | hf:<dataset>[:config][:split] | local .txt/.jsonl")
    p.add_argument("--n-seqs", type=int, default=n_seqs)
    p.add_argument("--seq-len", type=int, default=seq_len)
    p.add_argument("--batch-size", type=int, default=4)


def _dtype(s):
    return None if s is None else getattr(torch, s)


def _load_pair(args):
    from .hf import load_model, load_tokenizer, assert_shared_tokenizer
    kw = {"trust_remote_code": True} if args.trust_remote_code else {}
    src = load_model(args.source, device=args.source_device or args.device, dtype=_dtype(args.dtype),
                     attn_implementation=args.attn, **kw)
    tgt = load_model(args.target, device=args.target_device or args.device, dtype=_dtype(args.dtype),
                     attn_implementation=args.attn, **kw)
    tok = load_tokenizer(args.source, **kw)
    assert_shared_tokenizer(tok, load_tokenizer(args.target, **kw))
    return src, tgt, tok


def cmd_check(args):
    from transformers import AutoConfig
    from .hf import ModelSpec, check_matched_kv, load_tokenizer, assert_shared_tokenizer
    from .mapper import Mapper

    def spec_from_config(name):
        cfg = AutoConfig.from_pretrained(name, trust_remote_code=args.trust_remote_code)
        cfg = getattr(cfg, "text_config", None) or cfg
        hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        rp = getattr(cfg, "rope_parameters", None) or {}
        theta = rp.get("rope_theta", getattr(cfg, "rope_theta", float("nan")))
        return ModelSpec(name, int(cfg.num_hidden_layers), int(cfg.num_key_value_heads), int(hd), float(theta), 1.0)

    s, t = spec_from_config(args.source), spec_from_config(args.target)
    print("source:", s)
    print("target:", t)
    try:
        check_matched_kv(s, t)
        print("matched-KV: yes")
    except ValueError as e:
        print("matched-KV: NO --", e)
    try:
        assert_shared_tokenizer(load_tokenizer(args.source), load_tokenizer(args.target))
        print("shared tokenizer: yes")
    except Exception as e:  # noqa: BLE001
        print("shared tokenizer: NO --", e)
    for k in [int(x) for x in args.k.split(",")]:
        n = Mapper.formula_params(t, s, k)
        print(f"k={k}: mapper {n / 1e9:.2f} B params, {n * 4 / 1e9:.1f} GB fp32 / {n * 2 / 1e9:.1f} GB bf16")
    p, q = s.n_layers * s.kv_width, t.n_layers * t.kv_width
    print(f"calibration stats per kind (fp32): Gram {p * p * 4 / 1e9:.1f} GB + Cross {p * q * 4 / 1e9:.1f} GB")


def cmd_fit(args):
    from .calibration import calibrate, CalibrationStats
    from .data import batches, iter_dataset, token_sequences
    from .mapper import Mapper
    from .select import selection_score

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.stats and Path(args.stats, "meta.json").exists():
        print(f"loading calibration stats from {args.stats}")
        stats = CalibrationStats.load(args.stats, device=args.stats_device)
    else:
        src, tgt, tok = _load_pair(args)
        seqs = token_sequences(iter_dataset(args.data), tok, args.seq_len, args.n_seqs)
        print(f"calibrating on {len(seqs)} x {args.seq_len} tokens, stride {args.stride}")
        stats = calibrate(src, tgt, batches(seqs, args.batch_size), stride=args.stride,
                          stats_device=args.stats_device, stats_dtype=_dtype(args.stats_dtype) or torch.float32,
                          source_name=args.source, target_name=args.target,
                          require_matched_kv=not args.allow_mismatched, progress=True)
        if args.stats:
            stats.save(args.stats)
            print(f"saved calibration stats to {args.stats}")
    score = selection_score(stats)
    (out / "selection_r2.json").write_text(json.dumps({k: v.tolist() for k, v in score.items()}, indent=2))
    for k in args.k.split(","):
        kk = "all" if k == "all" else int(k)
        m = Mapper.fit(stats, k=kk, lam=args.lam, score=score["mean"], progress=args.verbose)
        d = out / f"k{m.k}"
        m.save(d, dtype=_dtype(args.save_dtype))
        print(f"saved {d}: {m.summary()}")


def cmd_eval(args):
    from .data import iter_dataset, token_sequences
    from .mapper import Mapper
    from .metrics import evaluate

    src, tgt, tok = _load_pair(args)
    m = Mapper.load(args.mapper)
    seqs = token_sequences(iter_dataset(args.data), tok, args.seq_len, args.n_seqs)
    rep = evaluate(src, tgt, m, seqs, prefix_len=args.seq_len - args.suffix_len, suffix_len=args.suffix_len, progress=True)
    print(rep.summary())
    if args.out:
        Path(args.out).write_text(json.dumps(rep.to_dict(), indent=2))


def cmd_bench(args):
    from .bench import benchmark, format_rows
    from .mapper import Mapper

    src, tgt, _ = _load_pair(args)
    m = Mapper.load(args.mapper)
    rows = benchmark(src, tgt, m, seq_lens=[int(x) for x in args.seq_lens.split(",")], warmup=args.warmup,
                     trials=args.trials, progress=True)
    print(format_rows(rows))


def cmd_generate(args):
    from .mapper import Mapper
    from .transfer import CrossModelTransfer

    src, tgt, tok = _load_pair(args)
    m = Mapper.load(args.mapper)
    xfer = CrossModelTransfer(src, tgt, m)
    ids = tok(args.prompt, return_tensors="pt")["input_ids"]
    res = xfer.generate(ids, max_new_tokens=args.max_new_tokens, hold_back=args.hold_back, eos_token_id=tok.eos_token_id)
    print("=== target from mapped cache ===")
    print(tok.decode(res.tokens[0, res.n_prompt:], skip_special_tokens=True))
    if args.compare:
        with torch.no_grad():
            own = tgt.generate(ids.to(next(tgt.parameters()).device), max_new_tokens=args.max_new_tokens, do_sample=False)
            s_own = src.generate(ids.to(next(src.parameters()).device), max_new_tokens=args.max_new_tokens, do_sample=False)
        print("=== target standalone ===")
        print(tok.decode(own[0, ids.shape[1]:], skip_special_tokens=True))
        print("=== source standalone ===")
        print(tok.decode(s_own[0, ids.shape[1]:], skip_special_tokens=True))


def cmd_inspect(args):
    from .mapper import Mapper
    m = Mapper.load(args.mapper)
    print(m.summary())
    print("selected source layers per target layer:")
    for lt, row in enumerate(m.selected):
        r2k = m.fit_r2.get("K", [float('nan')] * len(m.selected))[lt]
        r2v = m.fit_r2.get("V", [float('nan')] * len(m.selected))[lt]
        print(f"  target {lt:3d} <- {row.tolist()}  R2 K={r2k:.3f} V={r2v:.3f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kvtransfer", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="matched-KV / tokenizer check and mapper size for a pair")
    c.add_argument("--source", required=True)
    c.add_argument("--target", required=True)
    c.add_argument("--k", default="1,4,8,12,20")
    c.add_argument("--trust-remote-code", action="store_true")
    c.set_defaults(fn=cmd_check)

    f = sub.add_parser("fit", help="calibrate and fit mapper(s)")
    _add_pair(f)
    _add_data(f, 500, 1024)
    f.add_argument("--stride", type=int, default=4)
    f.add_argument("--k", default="8", help="comma list, e.g. 1,4,8,all")
    f.add_argument("--lam", type=float, default=0.01)
    f.add_argument("--out", required=True)
    f.add_argument("--stats", default=None, help="directory to save/load calibration statistics (reusable for any k)")
    f.add_argument("--stats-device", default=None)
    f.add_argument("--stats-dtype", default=None, help="float32 (default) or float64")
    f.add_argument("--save-dtype", default=None, help="store mapper weights as e.g. bfloat16 (default fp32)")
    f.add_argument("--allow-mismatched", action="store_true", help="skip the matched-KV check (untested regime)")
    f.add_argument("--verbose", action="store_true")
    f.set_defaults(fn=cmd_fit)

    e = sub.add_parser("eval", help="held-out R2, attention-output cosine, logit KL, top-1 agreement")
    _add_pair(e)
    _add_data(e, 32, 512)
    e.add_argument("--mapper", required=True)
    e.add_argument("--suffix-len", type=int, default=32)
    e.add_argument("--out", default=None, help="write the report as JSON")
    e.set_defaults(fn=cmd_eval)

    b = sub.add_parser("bench", help="mapper latency vs. target re-prefill")
    _add_pair(b)
    b.add_argument("--mapper", required=True)
    b.add_argument("--seq-lens", default="64,512,2048")
    b.add_argument("--warmup", type=int, default=5)
    b.add_argument("--trials", type=int, default=10)
    b.set_defaults(fn=cmd_bench)

    g = sub.add_parser("generate", help="decode on the target from the source's prefill")
    _add_pair(g)
    g.add_argument("--mapper", required=True)
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-new-tokens", type=int, default=64)
    g.add_argument("--hold-back", type=int, default=1)
    g.add_argument("--compare", action="store_true", help="also print source and target standalone generations")
    g.set_defaults(fn=cmd_generate)

    i = sub.add_parser("inspect", help="print a saved mapper's selection and fit R2")
    i.add_argument("--mapper", required=True)
    i.set_defaults(fn=cmd_inspect)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
