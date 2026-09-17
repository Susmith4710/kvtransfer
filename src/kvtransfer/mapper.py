"""The per-head linear mapper (paper Sec. 3, Fig. 3): fit from calibration moments, apply to a cache.

For each target layer ``l`` the mapper holds ``W_K[l]`` and ``W_V[l]`` of shape
``[k * n_kv * d_h, n_kv * d_h]`` plus biases.  Solving the multi-output ridge for all target
heads at once yields exactly the per-head solutions of Eq. 3 (the columns are independent), while
sharing the expensive Gram once per target layer, which is the paper's own trick (Appendix E).
Application is one batched matmul per target layer, with keys stripped of source RoPE before and
re-encoded with target RoPE after (Sec. 3.3).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .calibration import CalibrationStats
from .hf import ModelSpec, cache_layer, cache_num_layers, list_to_cache
from .rope import RopeCodec
from .select import selection_score, top_k_layers


@dataclass
class Mapper:
    source: ModelSpec
    target: ModelSpec
    selected: np.ndarray            # [L_t, k] source layers per target layer, best first
    lam: float
    W_K: list = field(default_factory=list)   # per target layer [k*kv_width, kv_width]
    b_K: list = field(default_factory=list)
    W_V: list = field(default_factory=list)
    b_V: list = field(default_factory=list)
    fit_r2: dict = field(default_factory=dict)  # {"K": [L_t], "V": [L_t]} in-sample per target layer
    meta: dict = field(default_factory=dict)
    key_space: str = "content"      # "content": strip source RoPE, map, re-apply target RoPE (paper Sec. 3.3)
                                    # "rope":    map the rotated keys directly (paper Table 2 "-all RoPE" ablation)

    # ---- properties ---------------------------------------------------------------------------
    @property
    def k(self) -> int:
        return int(self.selected.shape[1])

    def n_params(self, with_bias: bool = True) -> int:
        n = sum(int(w.numel()) for w in self.W_K) + sum(int(w.numel()) for w in self.W_V)
        if with_bias:
            n += sum(int(b.numel()) for b in self.b_K) + sum(int(b.numel()) for b in self.b_V)
        return n

    @staticmethod
    def formula_params(target: ModelSpec, source: ModelSpec, k: int) -> int:
        """Paper Appendix D: 2 * L_t * n_kv_t * (k * n_kv_s * d_h_s) * d_h_t (weights only)."""
        return 2 * target.n_layers * target.n_kv * (k * source.n_kv * source.head_dim) * target.head_dim

    def to(self, device=None, dtype=None) -> "Mapper":
        conv = lambda t: t.to(device=device, dtype=dtype)  # noqa: E731
        self.W_K = [conv(w) for w in self.W_K]
        self.b_K = [conv(b) for b in self.b_K]
        self.W_V = [conv(w) for w in self.W_V]
        self.b_V = [conv(b) for b in self.b_V]
        return self

    # ---- fitting ------------------------------------------------------------------------------
    @classmethod
    def fit(cls, stats: CalibrationStats, k="all", lam: float = 0.01, score: np.ndarray | None = None,
            selection_lam: float = 0.0, solve_dtype=torch.float64, progress: bool = False,
            key_space: str = "content") -> "Mapper":
        """Select top-k source layers per target layer and solve the per-layer ridge for K and V.

        ``key_space="rope"`` fits on the RoPE-coupled keys (calibration must include the ``Krope`` kind);
        selection always uses the content-space score as in the paper.
        """
        if key_space not in ("content", "rope"):  # "content-norerotate" only via ablate_inference_rope()
            raise ValueError("key_space must be 'content' or 'rope'")
        k_kind = "K" if key_space == "content" else "Krope"
        if score is None:
            score = selection_score(stats, selection_lam, kinds=("K", "V"))["mean"]
        selected = top_k_layers(score, k)
        m = cls(stats.source, stats.target, selected, lam, key_space=key_space,
                meta={"stride": stats.stride, "seq_len": stats.seq_len, "n_seqs": stats.n_seqs,
                      "n_tokens": {kind: acc.n for kind, acc in stats.acc.items()}})
        for kind in (k_kind, "V"):
            if kind not in stats.acc:
                raise ValueError(f"calibration stats lack kind {kind!r}; needed for key_space={key_space!r}")
        r2s = {"K": [], "V": []}
        for lt in range(stats.target.n_layers):
            rows = stats.src_rows(selected[lt].tolist())
            cols = stats.tgt_cols(lt)
            for kind in ("K", "V"):
                acc = stats.acc[k_kind if kind == "K" else "V"]
                w, b, _pooled = acc.solve(lam, rows=rows, cols=cols, solve_dtype=solve_dtype)
                (m.W_K if kind == "K" else m.W_V).append(w.float().cpu())
                (m.b_K if kind == "K" else m.b_V).append(b.float().cpu())
                # paper reports head-averaged R^2 (Table 7 / App. B), not pooled over the layer's columns
                per_head = acc.block_r2(*acc.last_column_residuals, block=stats.target.head_dim)
                r2s[kind].append(float(np.nanmean(per_head)))
            if progress:
                print(f"[fit] target layer {lt}: src {selected[lt].tolist()} R2 K={r2s['K'][-1]:.3f} V={r2s['V'][-1]:.3f}",
                      flush=True)
        m.fit_r2 = r2s
        m.meta["selection_score"] = score.tolist()
        return m

    # ---- application --------------------------------------------------------------------------
    @torch.no_grad()
    def apply_kv(self, src_kvs, positions: torch.Tensor, src_codec: RopeCodec, tgt_codec: RopeCodec,
                 out_dtype=None):
        """Map a list over source layers of ``(K_rope, V)`` [B, n_kv, T, d_h] to a list over target layers."""
        B, n_kv, T, d = src_kvs[0][0].shape
        dev = src_kvs[0][0].device
        wdt = self.W_K[0].dtype
        out_dtype = out_dtype or src_kvs[0][0].dtype
        needed = sorted(set(int(x) for x in self.selected.flatten()))
        # RoPE strip / re-apply always in fp32 (the paper's regime), the matmul in the mapper's dtype
        if self.key_space in ("content", "content-norerotate"):
            k_content = {l: src_codec.strip(src_kvs[l][0].float(), positions).to(wdt).permute(0, 2, 1, 3).reshape(B, T, n_kv * d)
                         for l in needed}
        else:
            k_content = {l: src_kvs[l][0].to(wdt).permute(0, 2, 1, 3).reshape(B, T, n_kv * d) for l in needed}
        v_src = {l: src_kvs[l][1].to(wdt).permute(0, 2, 1, 3).reshape(B, T, n_kv * d) for l in needed}
        tw, td = self.target.n_kv, self.target.head_dim
        out = []
        for lt in range(self.target.n_layers):
            layers = [int(x) for x in self.selected[lt]]
            xk = torch.cat([k_content[l] for l in layers], dim=-1)
            xv = torch.cat([v_src[l] for l in layers], dim=-1)
            k_hat = (xk @ self.W_K[lt].to(dev) + self.b_K[lt].to(dev)).reshape(B, T, tw, td).permute(0, 2, 1, 3)
            if self.key_space == "content":
                k_hat = tgt_codec.apply(k_hat.float(), positions)
            v_hat = (xv @ self.W_V[lt].to(dev) + self.b_V[lt].to(dev)).reshape(B, T, tw, td).permute(0, 2, 1, 3)
            out.append((k_hat.to(out_dtype).contiguous(), v_hat.to(out_dtype).contiguous()))
        return out

    @torch.no_grad()
    def apply_cache(self, src_cache, src_codec: RopeCodec, tgt_codec: RopeCodec, n_tokens: int | None = None,
                    target_model=None, out_dtype=None):
        """Map a source ``DynamicCache`` (first ``n_tokens`` positions) into a target ``DynamicCache``."""
        n_layers = cache_num_layers(src_cache)
        dev0 = cache_layer(src_cache, 0)[0].device
        if self.W_K and self.W_K[0].device != dev0:
            self.to(device=dev0)
        kvs = []
        for l in range(n_layers):
            k, v = cache_layer(src_cache, l)
            if n_tokens is not None:
                k, v = k[:, :, :n_tokens], v[:, :, :n_tokens]
            kvs.append((k, v))
        T = kvs[0][0].shape[2]
        positions = torch.arange(T, device=kvs[0][0].device)
        mapped = self.apply_kv(kvs, positions, src_codec, tgt_codec, out_dtype=out_dtype)
        return list_to_cache(mapped, target_model)

    # ---- (de)serialisation ------------------------------------------------------------------
    def save(self, path: str | Path, dtype=None) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        tensors = {}
        for l in range(len(self.W_K)):
            tensors[f"K.W.{l}"] = self.W_K[l].to(dtype) if dtype else self.W_K[l]
            tensors[f"K.b.{l}"] = self.b_K[l].to(dtype) if dtype else self.b_K[l]
            tensors[f"V.W.{l}"] = self.W_V[l].to(dtype) if dtype else self.W_V[l]
            tensors[f"V.b.{l}"] = self.b_V[l].to(dtype) if dtype else self.b_V[l]
        save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, str(path / "mapper.safetensors"))
        (path / "mapper.json").write_text(json.dumps({
            "format": "kvtransfer.mapper.v1",
            "paper": "arXiv:2608.03893",
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "k": self.k,
            "lam": self.lam,
            "key_space": self.key_space,
            "selected": self.selected.tolist(),
            "fit_r2": self.fit_r2,
            "n_params": self.n_params(),
            "meta": self.meta,
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path, device=None, dtype=None) -> "Mapper":
        path = Path(path)
        meta = json.loads((path / "mapper.json").read_text())
        t = load_file(str(path / "mapper.safetensors"))
        Lt = len(meta["selected"])
        m = cls(ModelSpec(**meta["source"]), ModelSpec(**meta["target"]), np.asarray(meta["selected"]),
                meta["lam"], fit_r2=meta.get("fit_r2", {}), meta=meta.get("meta", {}),
                key_space=meta.get("key_space", "content"))
        m.W_K = [t[f"K.W.{l}"] for l in range(Lt)]
        m.b_K = [t[f"K.b.{l}"] for l in range(Lt)]
        m.W_V = [t[f"V.W.{l}"] for l in range(Lt)]
        m.b_V = [t[f"V.b.{l}"] for l in range(Lt)]
        return m.to(device=device, dtype=dtype)

    def ablate_inference_rope(self) -> "Mapper":
        """Paper Table 2 "-inference RoPE": the content-space fit applied *without* re-rotating the mapped
        keys (a fit-vs-inference mismatch the paper shows collapses MMLU/GSM8K).  Shares weights."""
        m = Mapper(self.source, self.target, self.selected, self.lam, self.W_K, self.b_K, self.W_V, self.b_V,
                   self.fit_r2, dict(self.meta), key_space="content-norerotate")
        return m

    def summary(self) -> str:
        r2k = np.mean(self.fit_r2.get("K", [float("nan")]))
        r2v = np.mean(self.fit_r2.get("V", [float("nan")]))
        n = self.n_params()
        size = f"{n / 1e9:.2f} B params ({n * 4 / 1e9:.1f} GB fp32)" if n >= 1e8 else f"{n / 1e6:.2f} M params ({n * 4 / 1e6:.1f} MB fp32)"
        return (f"{self.source.name} -> {self.target.name}: k={self.k}, lambda={self.lam}, key_space={self.key_space}, "
                f"{size}, in-sample R2 K={r2k:.3f} V={r2v:.3f}")
