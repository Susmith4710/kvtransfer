"""Hardware profile, memory planner, and ``doctor`` for single-box targets such as the NVIDIA DGX Spark.

The DGX Spark (GB10 Grace Blackwell) is the reference target: 20 Arm cores + a Blackwell GPU
(compute capability 12.1) sharing one 128 GB LPDDR5x pool.  ``torch.cuda`` sees ~121.7 GB of it and
there is no separate VRAM, so GPU allocations compete with the OS and page cache, ``nvidia-smi``
reports memory as N/A, ``torch.cuda.mem_get_info()`` "free" tracks raw MemFree rather than
MemAvailable, and running above ~80 % of the pool has frozen boxes.  Everything below is sized with
that in mind: plan against a *usable* budget (80 % of the pool by default), never against "free".
"""
from __future__ import annotations

import math
import os
import platform
import shutil
from dataclasses import dataclass, field, asdict

GIB = 1024 ** 3


@dataclass
class HardwareProfile:
    name: str
    device: str                      # "cuda" | "cpu"
    device_name: str = ""
    compute_capability: str = ""
    gpu_visible_bytes: int = 0        # torch.cuda.mem_get_info()[1]
    system_total_bytes: int = 0       # /proc/meminfo MemTotal
    system_available_bytes: int = 0   # /proc/meminfo MemAvailable
    unified_memory: bool = False
    arch: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    bf16: bool = False
    flash_attn: bool = False
    recommended_attn: str = "sdpa"
    safe_fraction: float = 0.8
    notes: list = field(default_factory=list)

    @property
    def pool_bytes(self) -> int:
        """The memory pool models and statistics draw from."""
        if self.device == "cuda":
            return self.gpu_visible_bytes or self.system_total_bytes
        return self.system_total_bytes

    @property
    def usable_bytes(self) -> int:
        return int(self.pool_bytes * self.safe_fraction)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pool_gib"] = round(self.pool_bytes / GIB, 1)
        d["usable_gib"] = round(self.usable_bytes / GIB, 1)
        return d


# Offline preset so plans can be made on any machine before touching the Spark.
DGX_SPARK = HardwareProfile(
    name="dgx-spark", device="cuda", device_name="NVIDIA GB10", compute_capability="12.1",
    gpu_visible_bytes=int(121.7 * GIB), system_total_bytes=128 * GIB, system_available_bytes=int(115 * GIB),
    unified_memory=True, arch="aarch64", torch_version=">=2.9 (cu130)", cuda_version="13.0", bf16=True,
    flash_attn=False, recommended_attn="sdpa", safe_fraction=0.8,
    notes=[
        "Unified memory: GPU and CPU share the 128 GB pool; plan against 80 % of it, not against 'free'.",
        "nvidia-smi shows memory as N/A on GB10; use `free -h` (available) or this doctor.",
        "Install torch from the cu130 index (>= 2.9); cu12x wheels fail at import (only libcudart.so.13 exists).",
        "flash-attn has no sm_121 wheels; use attn_implementation='sdpa' (the default here).",
        "Set TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas before anything that uses torch.compile/Triton.",
        "Drop the page cache before big loads: sync; echo 3 | sudo tee /proc/sys/vm/drop_caches.",
        "Consider a cgroup MemoryMax (~100G) and disabling swap so an over-allocation fails instead of freezing the box.",
    ],
)

PRESETS = {"dgx-spark": DGX_SPARK}


def _meminfo() -> tuple[int, int]:
    total = avail = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
    except OSError:
        pass
    return total, avail


def detect() -> HardwareProfile:
    """Probe the current machine."""
    import torch

    total, avail = _meminfo()
    prof = HardwareProfile(name="local", device="cpu", arch=platform.machine(), torch_version=torch.__version__,
                           system_total_bytes=total, system_available_bytes=avail)
    if torch.cuda.is_available():
        prof.device = "cuda"
        prof.device_name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        prof.compute_capability = f"{cc[0]}.{cc[1]}"
        prof.cuda_version = str(torch.version.cuda)
        try:
            free, tot = torch.cuda.mem_get_info(0)
            prof.gpu_visible_bytes = int(tot)
        except Exception:  # noqa: BLE001
            prof.gpu_visible_bytes = int(torch.cuda.get_device_properties(0).total_memory)
        prof.bf16 = bool(torch.cuda.is_bf16_supported())
        name = prof.device_name.lower()
        if "gb10" in name or "spark" in name or ("orin" in name) or ("thor" in name):
            prof.unified_memory = True
            prof.name = "dgx-spark" if "gb10" in name else "unified"
            prof.notes.extend(DGX_SPARK.notes)
            # On unified memory, the GPU-visible total is the whole pool; be conservative.
            prof.safe_fraction = 0.8
        else:
            prof.safe_fraction = 0.9
        if shutil.which("nvidia-smi") is None:
            prof.notes.append("nvidia-smi not found on PATH")
    else:
        prof.bf16 = False
        prof.notes.append("No CUDA device: models will run on CPU in fp32 (fine for tiny tests only).")
    try:
        import flash_attn  # noqa: F401
        prof.flash_attn = True
        prof.recommended_attn = "sdpa" if prof.unified_memory else "flash_attention_2"
    except Exception:  # noqa: BLE001
        prof.flash_attn = False
        prof.recommended_attn = "sdpa"
    if prof.arch == "aarch64" and prof.device == "cuda" and "cu130" not in torch.__version__ and "+cu13" not in torch.__version__:
        prof.notes.append(f"torch {torch.__version__} on aarch64 CUDA: make sure it is a cu130 build.")
    if os.environ.get("TRITON_PTXAS_PATH") is None and prof.compute_capability.startswith("12."):
        prof.notes.append("TRITON_PTXAS_PATH is not set; torch.compile/Triton kernels may fail on sm_12x.")
    return prof


# ------------------------------------------------------------------------------------- planning

@dataclass
class ModelCost:
    name: str
    n_layers: int
    n_kv: int
    head_dim: int
    n_params: int
    weight_bytes: int              # as stored / as loaded
    vocab_size: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0

    @property
    def kv_width(self) -> int:
        return self.n_kv * self.head_dim

    def kv_cache_bytes(self, tokens: int, batch: int = 1, dtype_bytes: int = 2) -> int:
        return batch * tokens * self.n_layers * 2 * self.kv_width * dtype_bytes


def estimate_params(cfg: dict) -> int:
    """Analytic parameter count for Llama/Qwen/Mistral-style configs, dense or MoE."""
    h = int(cfg.get("hidden_size", 0))
    L = int(cfg.get("num_hidden_layers", 0))
    v = int(cfg.get("vocab_size", 0))
    nh = int(cfg.get("num_attention_heads", 1))
    nkv = int(cfg.get("num_key_value_heads", nh))
    dh = int(cfg.get("head_dim") or (h // max(nh, 1)))
    inter = int(cfg.get("intermediate_size", 0))
    tied = bool(cfg.get("tie_word_embeddings", False))
    attn = h * nh * dh + 2 * h * nkv * dh + nh * dh * h
    n_exp = int(cfg.get("num_experts") or cfg.get("num_local_experts") or 0)
    if n_exp:
        moe_inter = int(cfg.get("moe_intermediate_size") or inter)
        shared = int(cfg.get("shared_expert_intermediate_size") or 0)
        mlp = n_exp * 3 * h * moe_inter + 3 * h * shared + h * n_exp
    else:
        mlp = 3 * h * inter
    per_layer = attn + mlp + 2 * h
    embed = v * h * (1 if tied else 2)
    return int(L * per_layer + embed + h)


def dtype_bytes(dtype: str) -> int:
    return {"bfloat16": 2, "float16": 2, "float32": 4, "int8": 1, "fp8": 1, "int4": 0.5, "nvfp4": 0.5}.get(dtype, 2)


@dataclass
class PairPlan:
    source: ModelCost
    target: ModelCost
    dtype: str = "bfloat16"
    n_seqs: int = 500
    seq_len: int = 1024
    stride: int = 4
    batch_size: int = 4
    k_values: tuple = (1, 4, 8, 12, "all")
    stats_dtype_bytes: int = 4

    # ---- pieces --------------------------------------------------------------------------------
    @property
    def model_bytes(self) -> int:
        b = dtype_bytes(self.dtype)
        return int(self.source.n_params * b + self.target.n_params * b)

    @property
    def p(self) -> int:
        return self.source.n_layers * self.source.kv_width

    @property
    def q(self) -> int:
        return self.target.n_layers * self.target.kv_width

    def stats_bytes(self, kinds: int = 2) -> int:
        """Gram + Cross per kind (K, V) in the accumulator dtype."""
        return int(kinds * (self.p * self.p + self.p * self.q) * self.stats_dtype_bytes)

    def calibration_activation_bytes(self) -> int:
        """Rough peak transient memory of one calibration batch through both models (bf16 forward)."""
        B, T = self.batch_size, self.seq_len
        kv = self.source.kv_cache_bytes(T, B) + self.target.kv_cache_bytes(T, B)
        logits = B * T * max(self.source.vocab_size, self.target.vocab_size) * 4 * 2   # fp32 logits, both
        hidden = B * T * (max(self.source.hidden_size, self.target.hidden_size) * 8) * 2  # transient MLP/attn
        feats = (B * T // max(self.stride, 1)) * (self.p + self.q) * 4                    # extracted features
        return int(kv + logits + hidden + feats)

    def mapper_bytes(self, k, dtype_b: int = 4) -> int:
        kk = self.source.n_layers if k == "all" else int(k)
        return int(2 * self.target.n_layers * self.target.kv_width * (kk * self.source.kv_width) * dtype_b)

    def n_tokens(self) -> int:
        return self.n_seqs * self.seq_len // max(self.stride, 1)

    def fit_flops(self) -> float:
        """Forward passes of both models over the calibration set plus the Gram accumulation."""
        toks = self.n_seqs * self.seq_len
        fwd = 2.0 * (self.source.n_params + self.target.n_params) * toks
        gram = 2 * (2.0 * self.n_tokens() * (self.p ** 2 + self.p * self.q))
        return fwd + gram

    # ---- verdict -------------------------------------------------------------------------------
    def peak_bytes(self, kinds_per_pass: int = 2) -> int:
        return self.model_bytes + self.stats_bytes(kinds_per_pass) + self.calibration_activation_bytes()

    def fit(self, hw: HardwareProfile) -> dict:
        """Decide how (and whether) calibration fits in ``hw``'s usable budget."""
        budget = hw.usable_bytes
        one_pass = self.peak_bytes(2)
        two_pass = self.peak_bytes(1)
        models_only = self.model_bytes
        if one_pass <= budget:
            mode, peak = "single pass (K and V together)", one_pass
        elif two_pass <= budget:
            mode, peak = "two passes (K then V)", two_pass
        elif models_only + self.calibration_activation_bytes() <= budget:
            mode, peak = "does not fit with statistics on device; reduce seq_len/batch or use a smaller pair", two_pass
        else:
            mode, peak = "models alone exceed the budget", models_only
        fits = peak <= budget
        # crude time estimate; Spark bf16 dense sustains maybe ~50-100 TFLOPS in HF eager
        tflops = 60e12 if hw.device == "cuda" else 0.5e12
        return {
            "fits": fits,
            "mode": mode,
            "peak_gib": round(peak / GIB, 1),
            "budget_gib": round(budget / GIB, 1),
            "models_gib": round(self.model_bytes / GIB, 1),
            "stats_gib_per_kind": round(self.stats_bytes(1) / GIB, 1),
            "activations_gib": round(self.calibration_activation_bytes() / GIB, 1),
            "mappers": {str(k): round(self.mapper_bytes(k) / GIB, 2) for k in self.k_values},
            "calibration_tokens_per_head": self.n_tokens(),
            "rough_fit_minutes": round(self.fit_flops() / tflops / 60, 1),
            "recommended_batch_size": self.recommend_batch(hw),
        }

    def recommend_batch(self, hw: HardwareProfile) -> int:
        """Largest batch (1..8) whose two-pass peak stays within budget."""
        best = 1
        for b in (1, 2, 4, 8):
            trial = PairPlan(**{**asdict_shallow(self), "batch_size": b})
            if trial.peak_bytes(1) <= hw.usable_bytes:
                best = b
        return best


def asdict_shallow(plan: PairPlan) -> dict:
    return {
        "source": plan.source, "target": plan.target, "dtype": plan.dtype, "n_seqs": plan.n_seqs,
        "seq_len": plan.seq_len, "stride": plan.stride, "batch_size": plan.batch_size,
        "k_values": plan.k_values, "stats_dtype_bytes": plan.stats_dtype_bytes,
    }


def format_profile(p: HardwareProfile) -> str:
    lines = [
        f"profile        : {p.name}",
        f"device         : {p.device} {p.device_name} (cc {p.compute_capability})".rstrip(),
        f"arch / torch   : {p.arch} / torch {p.torch_version} / cuda {p.cuda_version}",
        f"memory pool    : {p.pool_bytes / GIB:.1f} GiB{' (unified CPU+GPU)' if p.unified_memory else ''}, "
        f"usable budget {p.usable_bytes / GIB:.1f} GiB ({int(p.safe_fraction * 100)} %)",
        f"system RAM     : total {p.system_total_bytes / GIB:.1f} GiB, available {p.system_available_bytes / GIB:.1f} GiB",
        f"bf16 / attn    : bf16={'yes' if p.bf16 else 'no'}, flash-attn={'installed' if p.flash_attn else 'no'}, "
        f"recommended attn_implementation='{p.recommended_attn}'",
    ]
    for n in p.notes:
        lines.append(f"  note: {n}")
    return "\n".join(lines)


def format_plan(plan: PairPlan, verdict: dict) -> str:
    s, t = plan.source, plan.target
    lines = [
        f"{s.name} ({s.n_layers}L, {s.n_kv}x{s.head_dim}, {s.n_params / 1e9:.1f}B) -> "
        f"{t.name} ({t.n_layers}L, {t.n_kv}x{t.head_dim}, {t.n_params / 1e9:.1f}B)",
        f"  calibration : {plan.n_seqs} x {plan.seq_len} tokens, stride {plan.stride}, batch {plan.batch_size} "
        f"-> {verdict['calibration_tokens_per_head']:,} tokens per head",
        f"  memory      : models {verdict['models_gib']} GiB + stats {verdict['stats_gib_per_kind']} GiB/kind + "
        f"activations {verdict['activations_gib']} GiB; peak {verdict['peak_gib']} GiB vs budget {verdict['budget_gib']} GiB",
        f"  verdict     : {'FITS' if verdict['fits'] else 'DOES NOT FIT'} - {verdict['mode']}; "
        f"recommended batch {verdict['recommended_batch_size']}; rough fit time {verdict['rough_fit_minutes']} min",
        f"  mappers     : " + ", ".join(f"k={k}: {g} GiB" for k, g in verdict["mappers"].items()),
    ]
    return "\n".join(lines)
