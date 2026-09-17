"""Find local checkpoints, decide which the method applies to, and enumerate matched-KV pairs.

Scans explicit directories, the Hugging Face hub cache (``$HF_HOME/hub`` or
``~/.cache/huggingface/hub``) and Ollama's store.  Ollama/GGUF models are listed only so you know
they exist: the method needs a PyTorch checkpoint (safetensors) because the KV cache has to be read
out of and written into a live model, which llama.cpp does not expose.  For each GGUF model the
matching Hugging Face id is suggested.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from .hardware import ModelCost, estimate_params

# Pairs the paper evaluated (Table 1), for annotating what is known.  Values: avg retention %.
PAPER_PAIRS = {
    ("Qwen3-14B", "Qwen3-32B"): "paper Tier 1: 97.6 % avg retention (k=8)",
    ("Qwen3-8B", "Qwen3-32B"): "paper Tier 1: 87.5 % avg retention (k=12)",
    ("Llama-3.1-8B", "Llama-3.1-70B"): "paper Tier 1: 72.8 % avg retention (k=20)",
    ("Ministral-3-3B", "Ministral-3-8B"): "paper Tier 1: 76.2 % avg retention (k=all)",
    ("Ministral-3-3B", "Ministral-3-14B"): "paper Tier 2: 44.2 % avg retention, ridge fails (MLP recovers)",
    ("Ministral-3-8B", "Ministral-3-14B"): "paper Tier 2: 41.6 % avg retention, ridge fails (MLP recovers)",
}

# Ollama tag -> Hugging Face id (bf16 safetensors), for the models a typical Ollama box holds.
OLLAMA_TO_HF = [
    (re.compile(r"^qwen2\.5:(\d+(?:\.\d+)?)b(?:-instruct)?"), lambda m: f"Qwen/Qwen2.5-{m.group(1)}B-Instruct"),
    (re.compile(r"^qwen2\.5-coder:(\d+(?:\.\d+)?)b"), lambda m: f"Qwen/Qwen2.5-Coder-{m.group(1)}B-Instruct"),
    (re.compile(r"^qwen3:30b-a3b-instruct-2507"), lambda m: "Qwen/Qwen3-30B-A3B-Instruct-2507"),
    (re.compile(r"^qwen3:30b-a3b"), lambda m: "Qwen/Qwen3-30B-A3B"),
    (re.compile(r"^qwen3:(\d+(?:\.\d+)?)b(?:-instruct)?(?:-2507)?"), lambda m: f"Qwen/Qwen3-{m.group(1)}B"),
    (re.compile(r"^llama3\.1:(\d+)b"), lambda m: f"meta-llama/Llama-3.1-{m.group(1)}B-Instruct"),
    (re.compile(r"^llama3\.2:(\d+)b"), lambda m: f"meta-llama/Llama-3.2-{m.group(1)}B-Instruct"),
    (re.compile(r"^llama3\.3:(\d+)b"), lambda m: f"meta-llama/Llama-3.3-{m.group(1)}B-Instruct"),
    (re.compile(r"^gemma3:(\d+)b"), lambda m: f"google/gemma-3-{m.group(1)}b-it"),
    (re.compile(r"^gemma4:(\d+)b"), lambda m: f"google/gemma-4-{m.group(1)}b-it"),
    (re.compile(r"^mistral:(\d+)b"), lambda m: f"mistralai/Mistral-{m.group(1)}B-Instruct-v0.3"),
    (re.compile(r"^ministral-3:(\d+)b"), lambda m: f"mistralai/Ministral-3-{m.group(1)}B-Instruct"),
]


@dataclass
class ModelInfo:
    name: str
    path: str
    model_type: str = ""
    architectures: list = field(default_factory=list)
    n_layers: int = 0
    n_heads: int = 0
    n_kv: int = 0
    head_dim: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    vocab_size: int = 0
    rope_theta: float = float("nan")
    rope_type: str = "default"
    tokenizer_fp: str = ""
    n_params: int = 0
    weight_bytes: int = 0
    torch_dtype: str = ""
    quantization: str = ""
    is_moe: bool = False
    support: str = "supported"          # "supported" | "supported (quantized)" | "unsupported: <reason>"
    notes: list = field(default_factory=list)

    @property
    def supported(self) -> bool:
        return self.support.startswith("supported")

    @property
    def family_key(self) -> str:
        """Models that can be paired must share a tokenizer; fall back to model_type+vocab if unknown."""
        return self.tokenizer_fp or f"{self.model_type}:{self.vocab_size}"

    def cost(self) -> ModelCost:
        return ModelCost(self.name, self.n_layers, self.n_kv, self.head_dim, self.n_params, self.weight_bytes,
                         self.vocab_size, self.hidden_size, self.intermediate_size)

    def to_dict(self) -> dict:
        return asdict(self) | {"supported": self.supported}


@dataclass
class OllamaModel:
    tag: str
    hf_suggestion: str | None
    size_bytes: int = 0


# ------------------------------------------------------------------------------------- scanning

def _hub_cache_dirs() -> list[Path]:
    home = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    return [Path(home)]


def _snapshot_dirs(root: Path) -> Iterable[tuple[str, Path]]:
    """Yield (name, dir) for every directory holding a config.json under ``root`` (hub cache aware)."""
    if not root.exists():
        return
    if (root / "config.json").exists():
        yield root.name, root
        return
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("models--"):
            snaps = d / "snapshots"
            if snaps.exists():
                cands = [s for s in sorted(snaps.iterdir()) if (s / "config.json").exists()]
                if cands:
                    name = d.name[len("models--"):].replace("--", "/")
                    yield name, sorted(cands, key=lambda s: s.stat().st_mtime)[-1]
        elif (d / "config.json").exists():
            yield d.name, d
        else:  # one level deeper (e.g. models/qwen/Qwen3-8B)
            for dd in sorted(d.iterdir()):
                if dd.is_dir() and (dd / "config.json").exists():
                    yield f"{d.name}/{dd.name}", dd


def _tokenizer_fingerprint(d: Path) -> str:
    for fn in ("tokenizer.json", "vocab.json", "tokenizer.model"):
        f = d / fn
        if f.exists():
            h = hashlib.sha256()
            with f.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()[:16]
    return ""


def _weight_bytes(d: Path) -> int:
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        try:
            meta = json.loads(idx.read_text()).get("metadata", {})
            if "total_size" in meta:
                return int(meta["total_size"])
        except (OSError, ValueError):
            pass
    total = 0
    for f in d.glob("*.safetensors"):
        try:
            total += f.stat().st_size
        except OSError:
            pass
    return total


def classify(cfg: dict, d: Path | None = None) -> tuple[str, list[str]]:
    """Support verdict for one config.  Mirrors the paper's scope: dense full attention, RoPE, matched KV."""
    notes: list[str] = []
    mt = str(cfg.get("model_type", ""))
    text = cfg.get("text_config") or cfg
    if d is not None and not any(d.glob("*.safetensors")) and not any(d.glob("*.bin")):
        if any(d.glob("*.gguf")):
            return "unsupported: GGUF weights (llama.cpp/Ollama); the method needs a PyTorch checkpoint", notes
        return "unsupported: no weights found next to config.json", notes
    if any(k in mt for k in ("mamba", "jamba", "nemotron_h", "qwen3_next", "falcon_h1", "zamba", "granitemoehybrid")):
        return f"unsupported: hybrid SSM/attention architecture ({mt}); paper scope is dense full attention", notes
    if "kv_lora_rank" in text or mt.startswith("deepseek_v"):
        return f"unsupported: MLA attention ({mt}) has no per-head K/V cache", notes
    layer_types = text.get("layer_types") or []
    if any("sliding" in str(x) for x in layer_types) or (text.get("sliding_window") and text.get("use_sliding_window", True)
                                                          and mt in ("gemma2", "gemma3", "gemma3_text", "gemma4", "gemma4_text")):
        return f"unsupported: sliding-window / local attention layers ({mt}); paper scope is dense full attention", notes
    if text.get("num_key_value_heads") is None:
        return "unsupported: config has no num_key_value_heads", notes
    if text.get("rope_theta") is None and not (text.get("rope_parameters") or text.get("rope_scaling") is not None):
        notes.append("no rope_theta in config; RoPE variant will be read from the model at runtime")
    if cfg.get("num_experts") or cfg.get("num_local_experts") or text.get("num_experts"):
        notes.append("MoE: KV cache is attention-only so the mapper applies, but the paper did not test MoE")
    q = cfg.get("quantization_config") or text.get("quantization_config")
    if q:
        method = str(q.get("quant_method", "quantized"))
        notes.append(f"{method}-quantized weights: KV is still computed in bf16, but the paper's numbers are bf16 weights")
        return f"supported (quantized: {method})", notes
    if mt in ("qwen2", "qwen3", "qwen3_moe", "llama", "mistral", "mistral3", "ministral3", "ministral", "gemma", "phi3", "olmo2"):
        return "supported", notes
    notes.append(f"model_type {mt!r} not on the tested list; supported if it is a dense RoPE decoder with a model-level rotary_emb")
    return "supported", notes


def inspect_dir(name: str, d: Path) -> ModelInfo:
    cfg = json.loads((d / "config.json").read_text())
    text = cfg.get("text_config") or cfg
    nh = int(text.get("num_attention_heads", 0) or 0)
    hidden = int(text.get("hidden_size", 0) or 0)
    dh = int(text.get("head_dim") or (hidden // nh if nh else 0))
    rp = text.get("rope_parameters") or text.get("rope_scaling") or {}
    theta = float(text.get("rope_theta") or rp.get("rope_theta") or float("nan"))
    support, notes = classify(cfg, d)
    info = ModelInfo(
        name=name, path=str(d), model_type=str(cfg.get("model_type", "")), architectures=list(cfg.get("architectures", [])),
        n_layers=int(text.get("num_hidden_layers", 0) or 0), n_heads=nh, n_kv=int(text.get("num_key_value_heads", 0) or 0),
        head_dim=dh, hidden_size=hidden, intermediate_size=int(text.get("intermediate_size", 0) or 0),
        vocab_size=int(text.get("vocab_size", 0) or 0), rope_theta=theta, rope_type=str(rp.get("rope_type") or rp.get("type") or "default"),
        tokenizer_fp=_tokenizer_fingerprint(d), n_params=estimate_params(text), weight_bytes=_weight_bytes(d),
        torch_dtype=str(cfg.get("torch_dtype") or cfg.get("dtype") or ""),
        quantization=str((cfg.get("quantization_config") or {}).get("quant_method", "")),
        is_moe=bool(text.get("num_experts") or text.get("num_local_experts")), support=support, notes=notes,
    )
    return info


def scan(roots: Iterable[str | Path] = (), include_hf_cache: bool = True) -> list[ModelInfo]:
    dirs: list[Path] = [Path(r).expanduser() for r in roots]
    if include_hf_cache:
        dirs += _hub_cache_dirs()
    seen, out = set(), []
    for root in dirs:
        for name, d in _snapshot_dirs(root):
            key = str(d.resolve())
            if key in seen:
                continue
            seen.add(key)
            try:
                out.append(inspect_dir(name, d))
            except Exception as e:  # noqa: BLE001
                out.append(ModelInfo(name=name, path=str(d), support=f"unsupported: could not parse config ({e})"))
    return out


def scan_ollama(roots: Iterable[str | Path] = ()) -> list[OllamaModel]:
    cands = [Path(r).expanduser() for r in roots] + [
        Path(os.environ.get("OLLAMA_MODELS", "")) if os.environ.get("OLLAMA_MODELS") else Path("/nonexistent"),
        Path("~/.ollama/models").expanduser(), Path("/usr/share/ollama/.ollama/models"), Path("/root/.ollama/models"),
    ]
    out = []
    for root in cands:
        man = root / "manifests"
        if not man.exists():
            continue
        for path in man.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(man).parts  # registry / library / name / tag
            if len(rel) < 4:
                continue
            tag = f"{rel[-2]}:{rel[-1]}"
            size = 0
            try:
                for layer in json.loads(path.read_text()).get("layers", []):
                    size += int(layer.get("size", 0))
            except (OSError, ValueError):
                pass
            out.append(OllamaModel(tag, hf_suggestion(tag), size))
    return out


def hf_suggestion(ollama_tag: str) -> str | None:
    t = ollama_tag.lower()
    for rx, fn in OLLAMA_TO_HF:
        m = rx.match(t)
        if m:
            return fn(m)
    return None


# ------------------------------------------------------------------------------------- pairing

@dataclass
class PairInfo:
    source: ModelInfo
    target: ModelInfo
    depth_ratio: float
    param_ratio: float
    direction: str            # "small->large" | "large->small"
    paper_note: str = ""

    def to_dict(self) -> dict:
        return {"source": self.source.name, "target": self.target.name, "depth_ratio": round(self.depth_ratio, 2),
                "param_ratio": round(self.param_ratio, 2), "direction": self.direction, "paper_note": self.paper_note}


def _paper_note(a: ModelInfo, b: ModelInfo) -> str:
    def canon(n: str) -> str:
        n = n.split("/")[-1]
        for suf in ("-Instruct-2507", "-Instruct", "-Base", "-it"):
            n = n.replace(suf, "")
        return n
    return PAPER_PAIRS.get((canon(a.name), canon(b.name)), "")


def enumerate_pairs(models: list[ModelInfo]) -> list[PairInfo]:
    """Ordered matched-KV pairs within each tokenizer family, closest depth first."""
    sup = [m for m in models if m.supported and m.n_layers and m.n_kv and m.head_dim]
    pairs = []
    for a in sup:
        for b in sup:
            if a is b or a.family_key != b.family_key:
                continue
            if (a.n_kv, a.head_dim) != (b.n_kv, b.head_dim):
                continue
            direction = "small->large" if a.n_params <= b.n_params else "large->small"
            pairs.append(PairInfo(a, b, b.n_layers / a.n_layers, (b.n_params / a.n_params) if a.n_params else float("nan"),
                                  direction, _paper_note(a, b) or _paper_note(b, a)))
    pairs.sort(key=lambda p: (p.direction != "small->large", abs(p.depth_ratio - 1.0), p.param_ratio))
    return pairs


def mismatched_kv_neighbours(models: list[ModelInfo]) -> list[tuple[ModelInfo, ModelInfo, str]]:
    """Same-family pairs the paper's method cannot use as-is, with the reason."""
    sup = [m for m in models if m.supported and m.n_layers]
    out = []
    for i, a in enumerate(sup):
        for b in sup[i + 1:]:
            if a.family_key != b.family_key:
                continue
            if (a.n_kv, a.head_dim) != (b.n_kv, b.head_dim):
                out.append((a, b, f"KV heads {a.n_kv}x{a.head_dim} vs {b.n_kv}x{b.head_dim} (mismatched-KV: untested by the paper)"))
    return out


# ------------------------------------------------------------------------------------- report

def format_models(models: list[ModelInfo]) -> str:
    lines = [f"{'model':48} {'type':10} {'L':>3} {'kv':>2}x{'dh':<4} {'params':>7} {'dtype':9} support"]
    for m in models:
        lines.append(f"{m.name[:48]:48} {m.model_type[:10]:10} {m.n_layers:>3} {m.n_kv:>2}x{m.head_dim:<4} "
                     f"{m.n_params / 1e9:>6.1f}B {m.torch_dtype[:9]:9} {m.support}")
        for n in m.notes:
            lines.append(f"{'':48}   note: {n}")
    return "\n".join(lines)


def format_pairs(pairs: list[PairInfo]) -> str:
    if not pairs:
        return "no matched-KV pairs found among supported models"
    lines = [f"{'source':36} {'target':36} {'dir':12} {'depth':>6} {'params':>7}  paper"]
    for p in pairs:
        lines.append(f"{p.source.name[:36]:36} {p.target.name[:36]:36} {p.direction:12} {p.depth_ratio:>5.2f}x "
                     f"{p.param_ratio:>6.2f}x  {p.paper_note}")
    return "\n".join(lines)


def format_ollama(models: list[OllamaModel]) -> str:
    if not models:
        return ""
    lines = ["Ollama (GGUF) models found - not usable directly; download the Hugging Face checkpoint instead:"]
    for m in models:
        lines.append(f"  {m.tag:45} {m.size_bytes / 1e9:>6.1f} GB  -> {m.hf_suggestion or '(no known HF equivalent)'}")
    return "\n".join(lines)
