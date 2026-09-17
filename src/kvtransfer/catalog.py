"""Offline catalog of model architectures so a fleet (e.g. an Ollama config) can be analysed for
transfer pairs before downloading anything.  Values are the published config.json numbers; the
runtime always re-reads the real config, so treat this as planning data.

``analyze_tags`` answers: for these models, which ordered pairs are paper-validated matched-KV
pairs, which are same-family matched-KV pairs the paper did not test, which are mismatched-KV
(research extension), which are cross-family with compatible tokenizers, and which models cannot be
used at all; plus which sibling checkpoints would give a paper-faithful pair.
"""
from __future__ import annotations

from dataclasses import dataclass

from .discover import PAPER_PAIRS, hf_suggestion
from .hardware import DGX_SPARK, GIB, HardwareProfile, ModelCost, PairPlan


@dataclass(frozen=True)
class CatalogEntry:
    hf_id: str
    family: str            # tokenizer family: models in one family share a tokenizer
    n_layers: int
    n_heads: int
    n_kv: int
    head_dim: int
    hidden: int
    intermediate: int
    n_params: float        # billions, total
    vocab: int
    moe: bool = False
    support: str = "supported"
    note: str = ""

    @property
    def short(self) -> str:
        return self.hf_id.split("/")[-1]

    def cost(self, dtype_bytes: int = 2) -> ModelCost:
        n = int(self.n_params * 1e9)
        return ModelCost(self.short, self.n_layers, self.n_kv, self.head_dim, n, n * dtype_bytes, self.vocab,
                         self.hidden, self.intermediate)


def _e(hf, fam, L, H, kv, dh, hid, inter, params, vocab=151936, **kw):
    return CatalogEntry(hf, fam, L, H, kv, dh, hid, inter, params, vocab, **kw)


CATALOG: list[CatalogEntry] = [
    # Qwen2.5 (shared BPE with Qwen3; the two tokenizers differ only in added special tokens)
    _e("Qwen/Qwen2.5-0.5B-Instruct", "qwen", 24, 14, 2, 64, 896, 4864, 0.49),
    _e("Qwen/Qwen2.5-1.5B-Instruct", "qwen", 28, 12, 2, 128, 1536, 8960, 1.54),
    _e("Qwen/Qwen2.5-3B-Instruct", "qwen", 36, 16, 2, 128, 2048, 11008, 3.09),
    _e("Qwen/Qwen2.5-7B-Instruct", "qwen", 28, 28, 4, 128, 3584, 18944, 7.62, vocab=152064),
    _e("Qwen/Qwen2.5-14B-Instruct", "qwen", 48, 40, 8, 128, 5120, 13824, 14.77, vocab=152064),
    _e("Qwen/Qwen2.5-32B-Instruct", "qwen", 64, 40, 8, 128, 5120, 27648, 32.76, vocab=152064),
    _e("Qwen/Qwen2.5-72B-Instruct", "qwen", 80, 64, 8, 128, 8192, 29568, 72.7, vocab=152064,
       support="unsupported: 145 GB in bf16 exceeds a 128 GB box"),
    # Qwen3 dense
    _e("Qwen/Qwen3-0.6B", "qwen", 28, 16, 8, 128, 1024, 3072, 0.60),
    _e("Qwen/Qwen3-1.7B", "qwen", 28, 16, 8, 128, 2048, 6144, 1.72),
    _e("Qwen/Qwen3-4B-Instruct-2507", "qwen", 36, 32, 8, 128, 2560, 9728, 4.02),
    _e("Qwen/Qwen3-8B", "qwen", 36, 32, 8, 128, 4096, 12288, 8.19),
    _e("Qwen/Qwen3-14B", "qwen", 40, 40, 8, 128, 5120, 17408, 14.77),
    _e("Qwen/Qwen3-32B", "qwen", 64, 64, 8, 128, 5120, 25600, 32.76),
    # Qwen3 MoE
    _e("Qwen/Qwen3-30B-A3B-Instruct-2507", "qwen", 48, 32, 4, 128, 2048, 6144, 30.5, moe=True,
       note="MoE (128 experts, 8 active); KV cache is attention-only so the mapper applies, untested by the paper"),
    # Llama 3.x (3.1 / 3.2 / 3.3 share the tokenizer)
    _e("meta-llama/Llama-3.2-1B-Instruct", "llama3", 16, 32, 8, 64, 2048, 8192, 1.24, vocab=128256),
    _e("meta-llama/Llama-3.2-3B-Instruct", "llama3", 28, 24, 8, 128, 3072, 8192, 3.21, vocab=128256),
    _e("meta-llama/Llama-3.1-8B-Instruct", "llama3", 32, 32, 8, 128, 4096, 14336, 8.03, vocab=128256),
    _e("meta-llama/Llama-3.1-70B-Instruct", "llama3", 80, 64, 8, 128, 8192, 28672, 70.6, vocab=128256,
       support="unsupported: 141 GB in bf16 exceeds a 128 GB box"),
    # Ministral 3 (multimodal wrapper around a Mistral decoder; layer counts from the paper's Appendix D)
    _e("mistralai/Ministral-3-3B-Instruct", "ministral3", 26, 32, 8, 128, 3072, 9216, 3.4, vocab=131072),
    _e("mistralai/Ministral-3-8B-Instruct", "ministral3", 34, 32, 8, 128, 4096, 12288, 8.4, vocab=131072),
    _e("mistralai/Ministral-3-14B-Instruct", "ministral3", 40, 32, 8, 128, 5120, 16384, 14.0, vocab=131072),
    # Gemma: local sliding-window layers -> outside the paper's dense full-attention scope
    _e("google/gemma-3-4b-it", "gemma3", 34, 8, 4, 256, 2560, 10240, 4.3, vocab=262208,
       support="unsupported: sliding-window local attention layers"),
    _e("google/gemma-3-12b-it", "gemma3", 48, 16, 8, 256, 3840, 15360, 12.2, vocab=262208,
       support="unsupported: sliding-window local attention layers"),
    _e("google/gemma-3-27b-it", "gemma3", 62, 32, 16, 128, 5376, 21504, 27.4, vocab=262208,
       support="unsupported: sliding-window local attention layers"),
    _e("google/gemma-4-26b-it", "gemma4", 0, 0, 0, 0, 0, 0, 26.0, vocab=262208,
       support="unsupported: Gemma family uses sliding-window local attention layers (verify config.json)"),
]

BY_ID = {e.hf_id: e for e in CATALOG}
BY_SHORT = {e.short.lower(): e for e in CATALOG}


def lookup(name: str) -> CatalogEntry | None:
    """Match an HF id, a short name, or an Ollama tag."""
    if name in BY_ID:
        return BY_ID[name]
    short = name.split("/")[-1].lower()
    if short in BY_SHORT:
        return BY_SHORT[short]
    for suf in ("-instruct-2507", "-instruct", "-it"):
        if short + suf in BY_SHORT:
            return BY_SHORT[short + suf]
    hf = hf_suggestion(name)
    if hf:
        return lookup(hf) if hf != name else None
    return None


@dataclass
class PairVerdict:
    source: CatalogEntry
    target: CatalogEntry
    category: str        # paper-validated | matched-kv | mismatched-kv | cross-family | unusable
    reason: str
    plan: dict | None = None

    def to_dict(self) -> dict:
        return {"source": self.source.hf_id, "target": self.target.hf_id, "category": self.category,
                "reason": self.reason, "plan": self.plan}


TOKENIZER_COMPAT = {("qwen", "qwen"): "identical-or-compatible"}


def classify_pair(a: CatalogEntry, b: CatalogEntry) -> tuple[str, str]:
    if not a.support.startswith("supported") or not b.support.startswith("supported"):
        return "unusable", (a.support if not a.support.startswith("supported") else b.support)
    if a.family != b.family:
        return "unusable", f"different tokenizers ({a.family} vs {b.family}); token positions cannot align"
    matched = (a.n_kv, a.head_dim) == (b.n_kv, b.head_dim)
    same_series = _series(a) == _series(b)
    note = PAPER_PAIRS.get((_canon(a), _canon(b)))
    if note:
        return "paper-validated", note
    if matched and same_series:
        return "matched-kv", "same family, matched KV heads and head dim (paper's regime, this pair not in the paper)"
    if matched:
        return "cross-family", (f"matched KV geometry but different model series ({a.short.split('-')[0]} vs "
                                f"{b.short.split('-')[0]}) sharing a tokenizer; the paper leaves cross-family transfer open")
    return "mismatched-kv", (f"KV {a.n_kv}x{a.head_dim} -> {b.n_kv}x{b.head_dim}; the ridge solve is defined "
                            f"but the paper never tested mismatched KV (research extension)")


def _series(e: CatalogEntry) -> str:
    """'Qwen2.5', 'Qwen3', 'Llama', 'Ministral' ... the architecture series within a tokenizer family."""
    return e.short.split("-")[0]


def _canon(e: CatalogEntry) -> str:
    n = e.short
    for suf in ("-Instruct-2507", "-Instruct", "-it"):
        n = n.replace(suf, "")
    return n


def analyze_tags(tags: list[str], hw: HardwareProfile = DGX_SPARK, n_seqs: int = 500, seq_len: int = 1024,
                 stride: int = 4, batch_size: int = 4) -> dict:
    entries, unknown = [], []
    for t in tags:
        e = lookup(t)
        (entries if e else unknown).append(e or t)
    uniq = []
    for e in entries:
        if e not in uniq:
            uniq.append(e)
    pairs: list[PairVerdict] = []
    for a in uniq:
        for b in uniq:
            if a is b:
                continue
            cat, reason = classify_pair(a, b)
            pv = PairVerdict(a, b, cat, reason)
            if cat != "unusable":
                plan = PairPlan(a.cost(), b.cost(), "bfloat16", n_seqs, seq_len, stride, batch_size)
                pv.plan = plan.fit(hw)
            pairs.append(pv)
    order = {"paper-validated": 0, "matched-kv": 1, "mismatched-kv": 2, "cross-family": 3, "unusable": 4}
    pairs.sort(key=lambda p: (order[p.category], p.source.n_params > p.target.n_params, p.target.n_params))
    # suggestions: for each supported model, catalog siblings that would form a matched-KV same-family pair
    suggestions = {}
    for e in uniq:
        if not e.support.startswith("supported"):
            continue
        sibs = [c for c in CATALOG if c is not e and c not in uniq and c.support.startswith("supported")
                and classify_pair(e, c)[0] in ("paper-validated", "matched-kv")]
        if sibs:
            suggestions[e.hf_id] = [c.hf_id for c in sibs]
    return {"models": uniq, "unknown": unknown, "pairs": pairs, "suggestions": suggestions, "hardware": hw.name}


def format_analysis(a: dict) -> str:
    L = [f"Hardware profile: {a['hardware']}", "", "Models:"]
    for e in a["models"]:
        geo = f"{e.n_layers}L {e.n_kv}x{e.head_dim} {e.n_params:.1f}B" if e.n_layers else "?"
        L.append(f"  {e.hf_id:42} {geo:22} {e.support}{('; ' + e.note) if e.note else ''}")
    for u in a["unknown"]:
        L.append(f"  {u:42} not in catalog; run `kvtransfer discover` on the downloaded checkpoint")
    L += ["", "Ordered pairs (source -> target):"]
    for p in a["pairs"]:
        if p.category == "unusable":
            continue
        fit = p.plan
        mem = f"peak {fit['peak_gib']} GiB / budget {fit['budget_gib']} GiB: {'fits' if fit['fits'] else 'NO'} ({fit['mode']})"
        L.append(f"  [{p.category:15}] {p.source.short:28} -> {p.target.short:28} {mem}")
        L.append(f"  {'':17} {p.reason}")
    unus = [p for p in a["pairs"] if p.category == "unusable"]
    if unus:
        L += ["", "Not usable:"]
        seen = set()
        for p in unus:
            key = tuple(sorted((p.source.hf_id, p.target.hf_id)))
            if key in seen:
                continue
            seen.add(key)
            L.append(f"  {p.source.short} <-> {p.target.short}: {p.reason}")
    if a["suggestions"]:
        L += ["", "To get a paper-faithful (same family, matched-KV) pair, add one of these siblings:"]
        for k, v in a["suggestions"].items():
            L.append(f"  {k}: " + ", ".join(s.split('/')[-1] for s in v))
    return "\n".join(L)
