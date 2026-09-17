"""RoPE stripping and re-application (paper Sec. 3.3, "content-space mapping").

The KV cache stores rotated keys ``k_rope(t) = R_theta(t) k_content``.  Because the rotation is
orthogonal, its inverse is the rotation by ``-theta``.  Hugging Face models use the "rotate-half"
layout and their rotary module may multiply cos/sin by an ``attention_scaling`` factor (YaRN and
friends), so the cached key is really ``m * R(t) k``.  ``RopeCodec`` reads cos/sin *from the model's
own rotary module* rather than re-deriving frequencies, so every RoPE variant transformers supports
(default, linear, dynamic, yarn, llama3, ...) is inverted exactly.
"""
from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x``: [..., T, d_h]; ``cos``/``sin``: broadcastable to it (typically [T, d_h] or [1, 1, T, d_h])."""
    return x * cos + rotate_half(x) * sin


def strip_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Exact inverse of :func:`apply_rope` for *unscaled* cos/sin (rotation by ``-theta``)."""
    return x * cos - rotate_half(x) * sin


class RopeCodec:
    """Strips / applies a specific model's RoPE using that model's rotary embedding module.

    ``cos_sin(positions)`` returns the *scaled* cos/sin exactly as the model uses them, shaped
    ``[1, 1, T, d_h]`` so they broadcast over ``[B, n_kv, T, d_h]`` cache tensors.
    """

    def __init__(self, rotary_module: torch.nn.Module, attention_scaling: float | None = None):
        self.rotary = rotary_module
        m = attention_scaling
        if m is None:
            m = float(getattr(rotary_module, "attention_scaling", 1.0))
        self.attention_scaling = m

    @classmethod
    def from_model(cls, model: torch.nn.Module) -> "RopeCodec":
        base = getattr(model, "model", model)
        rot = getattr(base, "rotary_emb", None)
        if rot is None:
            raise ValueError(
                "model exposes no `.model.rotary_emb`; only Llama/Qwen/Mistral-style HF models with a "
                "model-level rotary embedding are supported"
            )
        return cls(rot)

    @torch.no_grad()
    def cos_sin(self, positions: torch.Tensor, device=None, dtype=torch.float32):
        """Scaled cos/sin for integer ``positions`` [T] -> two tensors [1, 1, T, d_h]."""
        dev = device if device is not None else next(iter(self.rotary.buffers())).device
        pos = positions.to(dev).long().reshape(1, -1)
        probe = torch.zeros(1, 1, dtype=torch.float32, device=dev)
        cos, sin = self.rotary(probe, pos)  # [1, T, d_h], already times attention_scaling
        return cos[:, None].to(dtype), sin[:, None].to(dtype)

    @torch.no_grad()
    def apply(self, k_content: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """content-space K [B, n_kv, T, d_h] -> what this model writes into its cache."""
        cos, sin = self.cos_sin(positions, device=k_content.device, dtype=k_content.dtype)
        return apply_rope(k_content, cos, sin)  # scaling already inside cos/sin

    @torch.no_grad()
    def strip(self, k_rope: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """cached K [B, n_kv, T, d_h] -> content-space K.  Exact: R is orthogonal, so R^-1 = R(-t)."""
        cos, sin = self.cos_sin(positions, device=k_rope.device, dtype=k_rope.dtype)
        m2 = self.attention_scaling ** 2
        out = strip_rope(k_rope, cos, sin)
        return out / m2 if m2 != 1.0 else out
