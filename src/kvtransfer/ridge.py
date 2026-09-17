"""Closed-form centered ridge from streaming sufficient statistics (paper Eq. 3-4).

    W* = (Xc^T Xc + lambda I)^-1 Xc^T Yc,      b = mean(Y) - mean(X) W*

Nothing about the calibration tokens is stored.  For every (source feature block, target feature
block) we keep shifted second moments, so any sub-block of source layers can be solved later without
re-reading the data.  Shifting by the first batch's mean before accumulating removes the catastrophic
cancellation that plain raw moments suffer when means are large relative to variances, which is the
case for the outlier channels in Qwen-style keys.
"""
from __future__ import annotations

import torch


class MomentAccumulator:
    """Accumulates ``n``, ``sum(x-s)``, ``sum(y-s_y)``, ``(x-s)^T (x-s)``, ``(x-s)^T (y-s_y)``, ``diag((y-s_y)^T (y-s_y))``.

    ``x``: [n, p] source features, ``y``: [n, q] target features.  ``Gram`` is p x p, ``Cross`` is p x q.
    """

    def __init__(self, p: int, q: int, device=None, dtype=torch.float32):
        self.p, self.q = p, q
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.n = 0
        self.shift_x = None
        self.shift_y = None
        self.sum_x = torch.zeros(p, device=self.device, dtype=torch.float64)
        self.sum_y = torch.zeros(q, device=self.device, dtype=torch.float64)
        self.gram = torch.zeros(p, p, device=self.device, dtype=dtype)
        self.cross = torch.zeros(p, q, device=self.device, dtype=dtype)
        self.yy = torch.zeros(q, device=self.device, dtype=torch.float64)

    @torch.no_grad()
    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x = x.to(self.device, self.dtype)
        y = y.to(self.device, self.dtype)
        if self.shift_x is None:
            self.shift_x = x.mean(0)
            self.shift_y = y.mean(0)
        xs = x - self.shift_x
        ys = y - self.shift_y
        self.n += int(x.shape[0])
        self.sum_x += xs.sum(0).double()
        self.sum_y += ys.sum(0).double()
        self.gram.addmm_(xs.T, xs)
        self.cross.addmm_(xs.T, ys)
        self.yy += (ys.double() ** 2).sum(0)

    # ---- centered statistics -----------------------------------------------------------------
    def mean_x(self) -> torch.Tensor:
        return (self.sum_x / self.n).to(self.dtype) + self.shift_x

    def mean_y(self) -> torch.Tensor:
        return (self.sum_y / self.n).to(self.dtype) + self.shift_y

    def centered(self, rows: torch.Tensor | None = None, cols: torch.Tensor | None = None):
        """Return ``(Gxx, Gxy, syy, mx, my)`` centered, restricted to source ``rows`` / target ``cols``."""
        dx = (self.sum_x / self.n).to(self.dtype)  # mean of shifted x
        dy = (self.sum_y / self.n).to(self.dtype)
        if rows is not None:
            rows = rows.to(self.device)
        if cols is not None:
            cols = cols.to(self.device)
        g = self.gram if rows is None else self.gram[rows][:, rows]
        c = self.cross
        if rows is not None:
            c = c[rows]
        if cols is not None:
            c = c[:, cols]
        dxr = dx if rows is None else dx[rows]
        dyc = dy if cols is None else dy[cols]
        yy = self.yy if cols is None else self.yy[cols]
        gxx = g - self.n * torch.outer(dxr, dxr)
        gxy = c - self.n * torch.outer(dxr, dyc)
        syy = (yy - self.n * dyc.double() ** 2).to(self.dtype)  # per-column centered sum of squares
        mx = dxr + (self.shift_x if rows is None else self.shift_x[rows])
        my = dyc + (self.shift_y if cols is None else self.shift_y[cols])
        return gxx, gxy, syy, mx, my

    def solve(self, lam: float, rows=None, cols=None, solve_dtype=torch.float64):
        """Closed-form ridge on the selected block.  Returns ``(W [p_r, q_c], b [q_c], r2)``.

        ``r2`` is the in-sample pooled coefficient of determination of the fitted block,
        ``1 - SS_res / SS_tot`` (paper: head-averaged R^2 when the block is one head).
        """
        gxx, gxy, syy, mx, my = self.centered(rows, cols)
        gxx = gxx.to(solve_dtype)
        gxy = gxy.to(solve_dtype)
        a = gxx + lam * torch.eye(gxx.shape[0], device=gxx.device, dtype=solve_dtype)
        try:
            w = torch.linalg.solve(a, gxy)
        except RuntimeError:
            w = torch.linalg.lstsq(a, gxy).solution
        # per-column SS_res_c = syy_c - 2 (Wᵀ XcᵀYc)_cc + (Wᵀ XcᵀXc W)_cc ; pooled R^2 sums the columns
        ss_res_c = syy.double() - 2.0 * (w * gxy).sum(0) + (w * (gxx @ w)).sum(0)
        ss_tot = float(syy.double().sum())
        ss_res = float(ss_res_c.sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        self.last_column_residuals = (ss_res_c.cpu(), syy.double().cpu())
        w = w.to(self.dtype)
        b = my - mx @ w
        return w, b, r2

    @staticmethod
    def block_r2(ss_res_c: torch.Tensor, syy_c: torch.Tensor, block: int) -> list[float]:
        """R^2 per contiguous column block of width ``block`` (one block per head), from per-column sums."""
        out = []
        for i in range(0, ss_res_c.numel(), block):
            tot = float(syy_c[i:i + block].sum())
            out.append(1.0 - float(ss_res_c[i:i + block].sum()) / tot if tot > 0 else float("nan"))
        return out

    # ---- (de)serialisation ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "n": torch.tensor(self.n),
            "shift_x": self.shift_x.cpu(),
            "shift_y": self.shift_y.cpu(),
            "sum_x": self.sum_x.cpu(),
            "sum_y": self.sum_y.cpu(),
            "gram": self.gram.cpu(),
            "cross": self.cross.cpu(),
            "yy": self.yy.cpu(),
        }

    @classmethod
    def from_state_dict(cls, sd: dict, device=None) -> "MomentAccumulator":
        p, q = sd["cross"].shape
        acc = cls(p, q, device=device, dtype=sd["gram"].dtype)
        acc.n = int(sd["n"])
        acc.shift_x = sd["shift_x"].to(acc.device, acc.dtype)
        acc.shift_y = sd["shift_y"].to(acc.device, acc.dtype)
        acc.sum_x = sd["sum_x"].to(acc.device, torch.float64)
        acc.sum_y = sd["sum_y"].to(acc.device, torch.float64)
        acc.gram = sd["gram"].to(acc.device, acc.dtype)
        acc.cross = sd["cross"].to(acc.device, acc.dtype)
        acc.yy = sd["yy"].to(acc.device, torch.float64)
        return acc


def r2_score(y: torch.Tensor, y_hat: torch.Tensor) -> float:
    """Pooled R^2 over rows and columns (used for held-out evaluation)."""
    y = y.double()
    y_hat = y_hat.double()
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean(0)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if float(ss_tot) > 0 else float("nan")
