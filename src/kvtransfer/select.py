"""Cross-layer source selection (paper Sec. 2.3 probe and Sec. 3.2 top-k).

For every (source layer l', target layer l, head h) fit a single-source OLS map on the head's
d_h features (head-matched, Eq. 2) and record R^2.  Head-average, then average the RoPE-stripped
K and V heatmaps, and take the top-k source layers per target layer.
"""
from __future__ import annotations

import numpy as np
import torch

from .calibration import CalibrationStats


@torch.no_grad()
def probe_r2(stats: CalibrationStats, kind: str, lam: float = 0.0) -> np.ndarray:
    """Head-averaged single-source R^2 heatmap, shape [L_s, L_t] (rows = source layers).

    Matched-KV pairs use the paper's head-matched probe (source head h -> target head h, Eq. 2).
    For mismatched KV (different head count or head dim, untested by the paper) each target head is
    regressed on *all* source heads of the layer, which is the only well-defined per-head probe.
    """
    acc = stats.acc[kind]
    Ls, Lt, H = stats.source.n_layers, stats.target.n_layers, stats.target.n_kv
    matched = (stats.source.n_kv, stats.source.head_dim) == (stats.target.n_kv, stats.target.head_dim)
    out = np.zeros((Ls, Lt), dtype=np.float64)
    for ls in range(Ls):
        rows_all = stats.src_rows([ls])
        for lt in range(Lt):
            vals = []
            for h in range(H):
                rows = stats.src_rows_head(ls, h) if matched else rows_all
                _, _, r2 = acc.solve(lam, rows=rows, cols=stats.tgt_cols(lt, h))
                vals.append(r2)
            out[ls, lt] = float(np.mean(vals))
    return out


def selection_score(stats: CalibrationStats, lam: float = 0.0, kinds=("K", "V")) -> dict:
    """R^2 heatmaps for the given kinds plus their mean, the selection criterion of Sec. 3.2
    (head-averaged R^2 averaged over RoPE-stripped keys and values)."""
    maps = {kind: probe_r2(stats, kind, lam) for kind in kinds if kind in stats.acc}
    maps["mean"] = np.mean(np.stack(list(maps.values())), axis=0)
    return maps


def top_k_layers(score: np.ndarray, k) -> np.ndarray:
    """``score``: [L_s, L_t].  Returns [L_t, k] source-layer indices, best first.  ``k='all'`` keeps every layer."""
    Ls, Lt = score.shape
    kk = Ls if k in ("all", None) else int(k)
    if not 1 <= kk <= Ls:
        raise ValueError(f"k={k} out of range for {Ls} source layers")
    return np.stack([np.argsort(-score[:, lt], kind="stable")[:kk] for lt in range(Lt)])
