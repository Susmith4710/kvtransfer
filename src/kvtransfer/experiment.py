"""One command that runs the paper's protocol on a local model pair and writes a report.

Stages (each resumable from files under ``out_dir``):

1. **plan**       memory plan against the hardware profile (Sec. 3.1 sizes, Appendix D) -> ``plan.json``
2. **calibrate**  500 x 1,024 tokens, stride 4, K/V (and K_rope for the ablation) -> ``stats/``
3. **fit**        ridge, lambda 0.01, k sweep {1,2,4,6,8,10,12,16,20,24,all} clipped to L_s -> ``mappers/k*/``
4. **eval**       held-out R^2, attention-output cosine, logit KL, top-1 agreement per k -> ``eval_k*.json``;
                  best k = highest mean attention-output cosine.  This is *our* criterion: the paper selects k
                  by log-likelihood benchmark accuracy (Sec. 4.1, App. H) and presents cosine as the
                  cross-pair predictor of retention (Sec. 4.5); ``kvtransfer harness`` gives the paper's criterion.
5. **ablation**   at the best k, the rows of Table 2: full; "-inference RoPE" (content fit, no re-rotation);
                  "-all RoPE" (fit and apply on rotated keys); "-RoPE -cross-layer" (rotated keys, k=1);
                  "-RoPE -cross-layer -ridge" (rotated keys, k=1, lambda=0) -> ``ablation.json``
6. **reverse**    calibrate target->source and fit the reverse mapper at the best k, so large-to-small
                  transfer is evaluated (Sec. 4.2/4.5) and multi-turn can alternate -> ``reverse/``
7. **bench**      mapper vs. target re-prefill across ten sequence lengths 64..32768, 50 warmup + 30 timed
                  trials (App. G), with energy -> ``bench.json``
8. **multiturn**  drift over alternating handoffs on a long document (Sec. 4.6 analogue: KL / top-1 vs the
                  target's standalone distribution instead of CoQA F1) -> ``multiturn.json``
9. **report**     ``report.json`` + ``report.md``
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch

from . import hardware as hw
from .bench import benchmark
from .calibration import CalibrationStats, calibrate
from .data import batches, iter_dataset, token_sequences
from .energy import EnergyMeter
from .hf import cache_to_list, forward_with_cache, list_to_cache, load_pair, model_spec, prefill
from .mapper import Mapper
from .metrics import evaluate
from .rope import RopeCodec
from .select import selection_score

PAPER_K_SWEEP = (1, 2, 4, 6, 8, 10, 12, 16, 20, 24, "all")


@dataclass
class ExperimentConfig:
    source: str
    target: str
    out_dir: str
    data: str = "fineweb-edu"
    n_seqs: int = 500
    seq_len: int = 1024
    stride: int = 4
    batch_size: int = 4
    k_values: tuple = PAPER_K_SWEEP
    lam: float = 0.01
    eval_n_seqs: int = 32
    eval_seq_len: int = 1024
    eval_suffix_len: int = 32
    bench_seq_lens: tuple = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)   # App. G: ten lengths
    bench_warmup: int = 50                                                             # App. G
    bench_trials: int = 30                                                             # App. G
    multiturn_turns: int = 10
    multiturn_turn_tokens: int = 64
    ablation: bool = True
    reverse: bool = True                 # also calibrate/fit target->source (L->S eval + alternating multi-turn)
    device: str | None = None
    dtype: str | None = None
    attn: str | None = None
    hardware: str | None = None          # preset name for offline planning, else detect
    stats_device: str | None = None
    force: bool = False                  # run even if the plan says it does not fit
    allow_mismatched: bool = False       # mismatched-KV pair (research extension beyond the paper)
    trust_remote_code: bool = False
    stages: tuple = ("plan", "calibrate", "fit", "eval", "ablation", "reverse", "bench", "multiturn", "report")

    def to_dict(self) -> dict:
        return asdict(self)


def _dtype(s):
    return None if s is None else getattr(torch, s)


def _json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.to_dict() if hasattr(o, "to_dict") else str(o)))


def _load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _cost_from_model(model, name: str) -> hw.ModelCost:
    spec = model_spec(model, name)
    cfg = model.config
    text = getattr(cfg, "text_config", None) or cfg
    n_params = sum(p.numel() for p in model.parameters())
    b = next(model.parameters()).element_size()
    return hw.ModelCost(name, spec.n_layers, spec.n_kv, spec.head_dim, n_params, n_params * b,
                        int(getattr(text, "vocab_size", 0)), int(getattr(text, "hidden_size", 0)),
                        int(getattr(text, "intermediate_size", 0)))


def plan_from_configs(source: str, target: str, cfg: ExperimentConfig, profile: hw.HardwareProfile) -> dict:
    """Plan without loading weights (configs only)."""
    from transformers import AutoConfig
    costs = []
    for name in (source, target):
        c = AutoConfig.from_pretrained(name, trust_remote_code=cfg.trust_remote_code)
        text = getattr(c, "text_config", None) or c
        d = text.to_dict()
        nh = int(d.get("num_attention_heads", 1))
        dh = int(d.get("head_dim") or d["hidden_size"] // nh)
        n = hw.estimate_params(d)
        costs.append(hw.ModelCost(name, int(d["num_hidden_layers"]), int(d["num_key_value_heads"]), dh, n,
                                  n * hw.dtype_bytes(cfg.dtype or "bfloat16"), int(d.get("vocab_size", 0)),
                                  int(d.get("hidden_size", 0)), int(d.get("intermediate_size", 0))))
    plan = hw.PairPlan(costs[0], costs[1], dtype=cfg.dtype or ("bfloat16" if profile.device == "cuda" else "float32"),
                       n_seqs=cfg.n_seqs, seq_len=cfg.seq_len, stride=cfg.stride, batch_size=cfg.batch_size,
                       k_values=tuple(cfg.k_values))
    verdict = plan.fit(profile)
    return {"profile": profile.to_dict(), "verdict": verdict, "text": hw.format_plan(plan, verdict)}


@torch.no_grad()
def multiturn_drift(source_model, target_model, mapper_st: Mapper, ids: torch.Tensor, turns: int, turn_tokens: int,
                    mapper_ts: Mapper | None = None, hold_back: int = 1) -> dict:
    """Alternate the live model each turn (source, target, source, ...), mapping the whole cache at every
    switch, and compare the live model's next-token distributions on each turn's tokens with the
    target's standalone distributions.  Returns per-turn KL and top-1 agreement.  With only the
    source->target mapper, turns on the source are skipped from the comparison (drift is measured on
    the target's turns only); with both mappers the chain alternates fully like the paper's CoQA setup."""
    from .transfer import Session
    src_dev = next(source_model.parameters()).device
    tgt_dev = next(target_model.parameters()).device
    n = min(turns, ids.shape[1] // turn_tokens)
    chunks = [ids[:, i * turn_tokens:(i + 1) * turn_tokens] for i in range(n)]
    # target standalone, incremental
    own_cache, own_logits = None, []
    for c in chunks:
        if own_cache is None:
            out = target_model(input_ids=c.to(tgt_dev), use_cache=True)
        else:
            out = forward_with_cache(target_model, own_cache, c.to(tgt_dev), past_len=own_cache.get_seq_length())
        own_cache = out.past_key_values
        own_logits.append(torch.log_softmax(out.logits.float(), -1).cpu())
    mappers = {("s", "t"): mapper_st}
    if mapper_ts is not None:
        mappers[("t", "s")] = mapper_ts
    sess = Session({"s": source_model, "t": target_model}, mappers, start="s")
    rows = []
    live = "s"
    for t, c in enumerate(chunks):
        # alternate s, t, s, t ... when both mappers exist; otherwise s once, then t forever
        want = ("s" if t % 2 == 0 else "t") if mapper_ts is not None else ("s" if t == 0 else "t")
        if want != live and (live, want) in mappers:
            sess.switch_to(want, hold_back=hold_back)
            live = want
        logits = sess.feed(c)
        lp = torch.log_softmax(logits.float(), -1).cpu()
        if live == "t":
            kl = (own_logits[t].exp() * (own_logits[t] - lp)).sum(-1).mean()
            agree = (own_logits[t].argmax(-1) == lp.argmax(-1)).float().mean()
            rows.append({"turn": t, "live": live, "kl": float(kl), "top1_agreement": float(agree)})
        else:
            rows.append({"turn": t, "live": live, "kl": None, "top1_agreement": None})
    tk = [r for r in rows if r["kl"] is not None]
    slope = float(np.polyfit([r["turn"] for r in tk], [r["kl"] for r in tk], 1)[0]) if len(tk) >= 2 else None
    return {"turns": rows, "kl_slope_per_turn": slope, "turn_tokens": turn_tokens, "alternating": mapper_ts is not None}


def run_experiment(cfg: ExperimentConfig, log=print) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _json(out / "config.json", cfg.to_dict())
    profile = hw.PRESETS[cfg.hardware] if cfg.hardware else hw.detect()
    report: dict = {"source": cfg.source, "target": cfg.target, "profile": profile.to_dict(), "stages": {}}
    t_all = time.time()

    # ---- 1. plan ------------------------------------------------------------------------------
    if "plan" in cfg.stages:
        plan = plan_from_configs(cfg.source, cfg.target, cfg, profile)
        _json(out / "plan.json", plan)
        log(plan["text"])
        report["stages"]["plan"] = plan["verdict"]
        if not plan["verdict"]["fits"] and not cfg.force:
            log("plan says the pair does not fit; pass force=True / --force to try anyway")
            _json(out / "report.json", report)
            return report

    src = tgt = tok = None

    def models():
        nonlocal src, tgt, tok
        if src is None:
            src, tgt, tok = load_pair(cfg.source, cfg.target, device=cfg.device, dtype=_dtype(cfg.dtype),
                                      attn_implementation=cfg.attn, trust_remote_code=cfg.trust_remote_code)
        return src, tgt, tok

    seqs_cache: dict = {}

    def sequences(n: int, seq_len: int, skip: int = 0):
        key = (n, seq_len, skip)
        if key not in seqs_cache:
            _, _, tk = models()
            all_seqs = token_sequences(iter_dataset(cfg.data), tk, seq_len, n + skip)
            seqs_cache[key] = all_seqs[skip:]
        return seqs_cache[key]

    # ---- 2. calibrate -------------------------------------------------------------------------
    stats_dir = out / "stats"
    stats = None
    if "calibrate" in cfg.stages or "fit" in cfg.stages:
        if (stats_dir / "meta.json").exists():
            log(f"[calibrate] reusing {stats_dir}")
            stats = CalibrationStats.load(stats_dir, device=cfg.stats_device)
        else:
            s, t, _ = models()
            seqs = sequences(cfg.n_seqs, cfg.seq_len)
            kinds = ("K", "V", "Krope") if cfg.ablation else ("K", "V")
            t0 = time.time()
            with EnergyMeter() as em:
                stats = calibrate(s, t, batches(seqs, cfg.batch_size), stride=cfg.stride, kinds=kinds,
                                  stats_device=cfg.stats_device, source_name=cfg.source, target_name=cfg.target,
                                  require_matched_kv=not cfg.allow_mismatched, progress=True)
            stats.save(stats_dir)
            report["stages"]["calibrate"] = {"seconds": time.time() - t0, "n_seqs": stats.n_seqs, "tokens_per_head": stats.acc["K"].n,
                                             "energy": em.reading.to_dict() if em.reading else None}
            log(f"[calibrate] done in {time.time() - t0:.0f}s")

    # ---- 3. fit -------------------------------------------------------------------------------
    mappers: dict = {}
    if "fit" in cfg.stages:
        score_path = out / "selection_r2.json"
        if score_path.exists():
            score = {k: np.asarray(v) for k, v in json.loads(score_path.read_text()).items()}
        else:
            score = selection_score(stats)
            _json(score_path, {k: v.tolist() for k, v in score.items()})
        Ls = stats.source.n_layers
        ks = []
        for k in cfg.k_values:
            kk = Ls if k == "all" else int(k)
            if kk <= Ls and kk not in ks:
                ks.append(kk)
        fit_rows = []
        for kk in ks:
            d = out / "mappers" / f"k{kk}"
            if (d / "mapper.json").exists():
                m = Mapper.load(d)
            else:
                t0 = time.time()
                m = Mapper.fit(stats, k=kk, lam=cfg.lam, score=score["mean"])
                m.save(d)
                log(f"[fit] k={kk} in {time.time() - t0:.1f}s: {m.summary()}")
            mappers[kk] = m
            fit_rows.append({"k": kk, "n_params": m.n_params(), "gib_fp32": m.n_params() * 4 / hw.GIB,
                             "r2_K": float(np.mean(m.fit_r2["K"])), "r2_V": float(np.mean(m.fit_r2["V"]))})
        report["stages"]["fit"] = fit_rows
        _json(out / "fit.json", fit_rows)

    # ---- 4. eval ------------------------------------------------------------------------------
    best_k = None
    if "eval" in cfg.stages and mappers:
        s, t, _ = models()
        held = sequences(cfg.eval_n_seqs, cfg.eval_seq_len, skip=cfg.n_seqs)
        eval_rows = []
        for kk, m in mappers.items():
            p = out / f"eval_k{kk}.json"
            if p.exists():
                rep = json.loads(p.read_text())
            else:
                t0 = time.time()
                r = evaluate(s, t, m, held, prefix_len=cfg.eval_seq_len - cfg.eval_suffix_len, suffix_len=cfg.eval_suffix_len)
                rep = r.to_dict() | {"k": kk, "seconds": time.time() - t0}
                _json(p, rep)
                log(f"[eval] k={kk}: {r.summary().splitlines()[2].strip()} | KL mean {r.kl_mean:.3f} | top-1 {r.top1_agreement:.3f}")
            eval_rows.append(rep)
        valid = [r for r in eval_rows if r.get("attn_cosine_mean") is not None]
        best = max(valid, key=lambda r: r["attn_cosine_mean"]) if valid else None
        best_k = best["k"] if best else None
        report["stages"]["eval"] = {"per_k": [{k: r[k] for k in ("k", "attn_cosine_mean", "attn_cosine_min", "kl_mean", "kl_p95",
                                                                  "top1_agreement", "source_top1_agreement")}
                                              | {"r2_K": float(np.mean(r["r2_K"])), "r2_V": float(np.mean(r["r2_V"]))}
                                              for r in eval_rows],
                                    "best_k": best_k, "criterion": "max mean attention-output cosine"}
    if best_k is None and mappers:
        best_k = max(mappers)
    report["best_k"] = best_k

    # ---- 5. ablation --------------------------------------------------------------------------
    if "ablation" in cfg.stages and cfg.ablation and best_k is not None and stats is not None and "Krope" in stats.acc:
        s, t, _ = models()
        held = sequences(cfg.eval_n_seqs, cfg.eval_seq_len, skip=cfg.n_seqs)
        p = out / "ablation.json"
        if p.exists():
            abl = json.loads(p.read_text())
        else:
            score = {k: np.asarray(v) for k, v in json.loads((out / "selection_r2.json").read_text()).items()}
            # Table 2 rows, removed sequentially as in the paper
            variants = {
                "full (content-space)": mappers[best_k],
                "-inference RoPE (content fit, no re-rotation)": mappers[best_k].ablate_inference_rope(),
                "-all RoPE (fit+apply on rotated keys)": Mapper.fit(stats, k=best_k, lam=cfg.lam, score=score["mean"], key_space="rope"),
                "-RoPE -cross-layer (rotated keys, k=1)": Mapper.fit(stats, k=1, lam=cfg.lam, score=score["mean"], key_space="rope"),
                "-RoPE -cross-layer -ridge (rotated keys, k=1, lambda=0)": Mapper.fit(stats, k=1, lam=0.0, score=score["mean"], key_space="rope"),
            }
            abl = {}
            for name, m in variants.items():
                r = evaluate(s, t, m, held, prefix_len=cfg.eval_seq_len - cfg.eval_suffix_len, suffix_len=cfg.eval_suffix_len)
                abl[name] = {"attn_cosine_mean": r.attn_cosine_mean, "kl_mean": r.kl_mean, "top1_agreement": r.top1_agreement,
                             "r2_K": float(np.nanmean(r.r2_K)), "r2_V": float(np.nanmean(r.r2_V))}
                log(f"[ablation] {name}: cosine {r.attn_cosine_mean:.3f} KL {r.kl_mean:.3f} top-1 {r.top1_agreement:.3f}")
            _json(p, abl)
        report["stages"]["ablation"] = abl

    # ---- 6. reverse direction ------------------------------------------------------------------
    reverse_mapper = None
    if "reverse" in cfg.stages and cfg.reverse and best_k is not None:
        s, t, _ = models()
        rdir = out / "reverse"
        rstats_dir = rdir / "stats"
        if (rdir / "mapper" / "mapper.json").exists():
            reverse_mapper = Mapper.load(rdir / "mapper")
            rev = json.loads((rdir / "eval.json").read_text()) if (rdir / "eval.json").exists() else {}
        else:
            if (rstats_dir / "meta.json").exists():
                rstats = CalibrationStats.load(rstats_dir, device=cfg.stats_device)
            else:
                t0 = time.time()
                rstats = calibrate(t, s, batches(sequences(cfg.n_seqs, cfg.seq_len), cfg.batch_size), stride=cfg.stride,
                                   kinds=("K", "V"), stats_device=cfg.stats_device, source_name=cfg.target, target_name=cfg.source,
                                   require_matched_kv=not cfg.allow_mismatched, progress=True)
                rstats.save(rstats_dir)
                log(f"[reverse] calibrated target->source in {time.time() - t0:.0f}s")
            k_rev = min(best_k, rstats.source.n_layers)
            reverse_mapper = Mapper.fit(rstats, k=k_rev, lam=cfg.lam)
            reverse_mapper.save(rdir / "mapper")
            held = sequences(cfg.eval_n_seqs, cfg.eval_seq_len, skip=cfg.n_seqs)
            r = evaluate(t, s, reverse_mapper, held, prefix_len=cfg.eval_seq_len - cfg.eval_suffix_len, suffix_len=cfg.eval_suffix_len)
            rev = r.to_dict() | {"k": k_rev, "direction": f"{cfg.target} -> {cfg.source}"}
            _json(rdir / "eval.json", rev)
            log(f"[reverse] L->S k={k_rev}: cosine {r.attn_cosine_mean:.3f} KL {r.kl_mean:.3f} top-1 {r.top1_agreement:.3f}")
            del rstats
        report["stages"]["reverse"] = {k: rev.get(k) for k in ("k", "direction", "attn_cosine_mean", "attn_cosine_min",
                                                                 "kl_mean", "kl_p95", "top1_agreement")} if rev else {}

    # ---- 7. bench -----------------------------------------------------------------------------
    if "bench" in cfg.stages and best_k is not None:
        s, t, _ = models()
        p = out / "bench.json"
        if p.exists():
            rows = json.loads(p.read_text())
        else:
            max_pos = int(getattr(getattr(t.config, "text_config", None) or t.config, "max_position_embeddings", 32768))
            lens = [n for n in cfg.bench_seq_lens if n <= max_pos]
            rows = []
            for n in lens:
                with EnergyMeter() as em_m:
                    r = benchmark(s, t, mappers[best_k], seq_lens=(n,), warmup=cfg.bench_warmup, trials=cfg.bench_trials)[0]
                rows.append({"seq_len": n, "mapper_ms": r.mapper_ms, "reprefill_ms": r.reprefill_ms, "speedup": r.speedup,
                             "energy_both": em_m.reading.to_dict() if em_m.reading else None})
                log(f"[bench] T={n}: mapper {r.mapper_ms:.1f} ms vs re-prefill {r.reprefill_ms:.1f} ms ({r.speedup:.1f}x)")
            _json(p, rows)
        report["stages"]["bench"] = rows

    # ---- 8. multiturn -------------------------------------------------------------------------
    if "multiturn" in cfg.stages and best_k is not None:
        s, t, _ = models()
        p = out / "multiturn.json"
        if p.exists():
            mt = json.loads(p.read_text())
        else:
            need = cfg.multiturn_turns * cfg.multiturn_turn_tokens
            doc = sequences(1, need, skip=cfg.n_seqs + cfg.eval_n_seqs)[0][None]
            mt = multiturn_drift(s, t, mappers[best_k], doc, cfg.multiturn_turns, cfg.multiturn_turn_tokens,
                                 mapper_ts=reverse_mapper)
            _json(p, mt)
            log(f"[multiturn] KL slope {mt['kl_slope_per_turn']} per turn")
        report["stages"]["multiturn"] = mt

    # ---- 9. report ----------------------------------------------------------------------------
    report["seconds_total"] = time.time() - t_all
    _json(out / "report.json", report)
    (out / "report.md").write_text(format_report(report, cfg))
    log(f"[report] {out / 'report.md'}")
    return report


def format_report(rep: dict, cfg: ExperimentConfig) -> str:
    L = [f"# kvtransfer experiment: {rep['source']} -> {rep['target']}", ""]
    prof = rep.get("profile", {})
    L.append(f"Hardware: {prof.get('name')} {prof.get('device_name', '')} ({prof.get('pool_gib')} GiB pool). "
             f"Calibration {cfg.n_seqs} x {cfg.seq_len} tokens, stride {cfg.stride}, lambda {cfg.lam}. Best k = {rep.get('best_k')}.")
    st = rep.get("stages", {})
    if "plan" in st:
        v = st["plan"]
        L += ["", "## Plan", f"- {v['mode']}; peak {v['peak_gib']} GiB vs budget {v['budget_gib']} GiB; rough fit time {v['rough_fit_minutes']} min"]
    if "fit" in st:
        L += ["", "## Fit (in-sample R2, paper Table 7 analogue)", "| k | params | GiB fp32 | R2 K | R2 V |", "|---|---:|---:|---:|---:|"]
        for r in st["fit"]:
            L.append(f"| {r['k']} | {r['n_params'] / 1e9:.2f} B | {r['gib_fp32']:.1f} | {r['r2_K']:.3f} | {r['r2_V']:.3f} |")
    if "eval" in st:
        L += ["", "## Held-out diagnostics (paper Sec. 4.5)", "| k | attn cosine | min layer | KL mean | KL p95 | top-1 | held-out R2 K | R2 V |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in st["eval"]["per_k"]:
            L.append(f"| {r['k']} | {r['attn_cosine_mean']:.3f} | {r['attn_cosine_min']:.3f} | {r['kl_mean']:.3f} | {r['kl_p95']:.3f} | "
                     f"{r['top1_agreement']:.3f} | {r['r2_K']:.3f} | {r['r2_V']:.3f} |")
        L.append(f"\nBest k by {st['eval']['criterion']} (our criterion; the paper selects k by benchmark accuracy): "
                 f"{st['eval']['best_k']}. Source-vs-target next-token agreement (no transfer): "
                 f"{st['eval']['per_k'][0]['source_top1_agreement']:.3f}.")
    if "ablation" in st:
        L += ["", "## Ablation at best k (paper Table 2 analogue)", "| variant | attn cosine | KL mean | top-1 | R2 K | R2 V |", "|---|---:|---:|---:|---:|---:|"]
        for name, r in st["ablation"].items():
            L.append(f"| {name} | {r['attn_cosine_mean']:.3f} | {r['kl_mean']:.3f} | {r['top1_agreement']:.3f} | {r['r2_K']:.3f} | {r['r2_V']:.3f} |")
    if st.get("reverse"):
        r = st["reverse"]
        L += ["", "## Large-to-small direction (paper Sec. 4.2 / 4.5, HellaSwag-only there)",
              f"{r['direction']} at k={r['k']}: attn cosine {r['attn_cosine_mean']:.3f} (min layer {r['attn_cosine_min']:.3f}), "
              f"KL mean {r['kl_mean']:.3f} (p95 {r['kl_p95']:.3f}), top-1 {r['top1_agreement']:.3f}"]
    if "bench" in st:
        L += ["", "## Latency: mapper vs. target re-prefill (paper Sec. 4.7)", "| seq len | mapper ms | re-prefill ms | speedup | energy method |", "|---:|---:|---:|---:|---|"]
        for r in st["bench"]:
            e = (r.get("energy_both") or {}).get("method", "n/a")
            L.append(f"| {r['seq_len']} | {r['mapper_ms']:.1f} | {r['reprefill_ms']:.1f} | {r['speedup']:.1f}x | {e} |")
    if "multiturn" in st:
        mt = st["multiturn"]
        mode = "alternating source/target every turn" if mt.get("alternating") else "one handoff source->target, then target only"
        L += ["", "## Multi-turn drift (paper Sec. 4.6 analogue)", f"Mode: {mode}. KL slope per turn: {mt.get('kl_slope_per_turn')}", "",
              "| turn | live | KL vs target standalone | top-1 |", "|---:|---|---:|---:|"]
        for r in mt["turns"]:
            kl = "" if r["kl"] is None else f"{r['kl']:.3f}"
            ag = "" if r["top1_agreement"] is None else f"{r['top1_agreement']:.3f}"
            L.append(f"| {r['turn']} | {r['live']} | {kl} | {ag} |")
    L += ["", "## How to read this", "- Attention-output cosine is the paper's cross-pair retention predictor (r=+0.57); R2 is not (r=-0.20).",
          "- Downstream accuracy retention (ARC/HellaSwag/MMLU/GSM8K) is what the paper reports; run `kvtransfer harness` for it.",
          "- The paper's Tier 1 pairs reach cosine-driven retention of 73-98 %; a pair with low cosine and high KL is a Tier 2 case where ridge fails."]
    return "\n".join(L)
