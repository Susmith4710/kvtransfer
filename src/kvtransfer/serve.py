"""A small HTTP service that mirrors an escalation cascade: a small model answers, and when the
request is escalated the large model continues from the small model's KV cache instead of
re-prefilling.

This is the shape of a memory-first pod's SYNTHESIZE -> ESCALATE path.  Per session the service
keeps the small model's token sequence and cache; on escalation it maps the **longest common token
prefix** between what the small model saw and the escalation prompt (system prompt + briefing are
usually identical, the instruction tail may differ), and the large model prefills only the
remainder.  Every response reports how many tokens were skipped, the mapper time, and the energy.

Endpoints (JSON):
  GET  /health
  POST /v1/generate   {"session": "id", "role": "source"|"target"|"escalate",
                       "prompt": "..."  |  "messages": [...],  "chat": true,
                       "max_new_tokens": 64, "hold_back": 1, "baseline": false}
        role=source    : small model answers; its cache is kept for the session
        role=escalate  : large model continues from the mapped cache (or re-prefills with baseline=true)
        role=target    : large model standalone (reference)
  POST /v1/reset      {"session": "id"}

Single-threaded on purpose: one GPU, models resident, greedy decoding.  Not a production server.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

from .energy import EnergyMeter
from .hf import cache_to_list, crop_cache, encode_prompt, forward_with_cache, list_to_cache, prefill
from .mapper import Mapper
from .transfer import CrossModelTransfer


def common_prefix_len(a: torch.Tensor, b: torch.Tensor) -> int:
    n = min(a.shape[-1], b.shape[-1])
    if n == 0:
        return 0
    eq = (a[..., :n] == b[..., :n]).reshape(-1)
    nz = (~eq).nonzero()
    return int(nz[0]) if nz.numel() else n


@dataclass
class SessionState:
    source_tokens: torch.Tensor | None = None     # [1, T] tokens the source has in cache
    source_cache: object = None
    history: list = field(default_factory=list)


class Escalator:
    """Holds the pair and the mapper; does the three kinds of generation with accounting."""

    def __init__(self, source_model, target_model, mapper: Mapper, tokenizer, chat_default: bool = True):
        self.xfer = CrossModelTransfer(source_model, target_model, mapper)
        self.tok = tokenizer
        self.chat_default = chat_default
        self.sessions: dict[str, SessionState] = {}
        self.eos = tokenizer.eos_token_id

    def _ids(self, req: dict) -> torch.Tensor:
        chat = bool(req.get("chat", self.chat_default))
        if "messages" in req:
            return encode_prompt(self.tok, req["messages"], chat=True)
        return encode_prompt(self.tok, req["prompt"], chat=chat, system=req.get("system"),
                             enable_thinking=req.get("enable_thinking"))

    @torch.no_grad()
    def _decode(self, model, cache, logits, seq: torch.Tensor, max_new: int):
        dev = next(model.parameters()).device
        new = []
        for _ in range(max_new):
            nxt = logits[:, -1].argmax(-1)
            new.append(int(nxt))
            if self.eos is not None and int(nxt) == self.eos:
                break
            out = forward_with_cache(model, cache, nxt[:, None].to(dev), past_len=seq.shape[1] + len(new) - 1)
            logits, cache = out.logits, out.past_key_values
        return new, cache

    @torch.no_grad()
    def generate(self, req: dict) -> dict:
        role = req.get("role", "source")
        sid = str(req.get("session", "default"))
        max_new = int(req.get("max_new_tokens", 64))
        hold_back = int(req.get("hold_back", 1))
        st = self.sessions.setdefault(sid, SessionState())
        ids = self._ids(req)
        T = ids.shape[1]
        timing: dict = {}
        with EnergyMeter() as em:
            t0 = time.perf_counter()
            if role == "source":
                model = self.xfer.source
                ids_d = ids.to(self.xfer.src_dev)
                logits, cache = prefill(model, ids_d)
                timing["prefill_ms"] = (time.perf_counter() - t0) * 1000
                timing["prefill_tokens"] = T
                t1 = time.perf_counter()
                new, cache = self._decode(model, cache, logits, ids_d, max_new)
                timing["decode_ms"] = (time.perf_counter() - t1) * 1000
                st.source_tokens = torch.cat([ids_d, torch.tensor([new], device=ids_d.device)], dim=1)
                st.source_cache = cache
            elif role == "target" or (role == "escalate" and (req.get("baseline") or st.source_cache is None)):
                model = self.xfer.target
                ids_d = ids.to(self.xfer.tgt_dev)
                logits, cache = prefill(model, ids_d)
                timing["prefill_ms"] = (time.perf_counter() - t0) * 1000
                timing["prefill_tokens"] = T
                timing["skipped_tokens"] = 0
                t1 = time.perf_counter()
                new, cache = self._decode(model, cache, logits, ids_d, max_new)
                timing["decode_ms"] = (time.perf_counter() - t1) * 1000
                if role == "escalate":
                    timing["mode"] = "re-prefill baseline" if req.get("baseline") else "re-prefill (no source cache for session)"
            elif role == "escalate":
                model = self.xfer.target
                ids_s = ids.to(self.xfer.src_dev)
                shared = common_prefix_len(st.source_tokens, ids_s)
                n_map = max(0, min(shared, T - hold_back))
                if n_map == 0:
                    logits, cache = prefill(model, ids.to(self.xfer.tgt_dev))
                    timing["mode"] = "re-prefill (no common prefix)"
                    timing["skipped_tokens"] = 0
                else:
                    t_m = time.perf_counter()
                    tgt_cache = self.xfer.map_cache(st.source_cache, n_map)
                    timing["mapper_ms"] = (time.perf_counter() - t_m) * 1000
                    tail = ids[:, n_map:].to(self.xfer.tgt_dev)
                    out = forward_with_cache(model, tgt_cache, tail, past_len=n_map)
                    logits, cache = out.logits, out.past_key_values
                    timing["mode"] = "transfer"
                    timing["skipped_tokens"] = n_map
                    timing["tail_tokens"] = int(tail.shape[1])
                timing["prefill_ms"] = (time.perf_counter() - t0) * 1000
                timing["prefill_tokens"] = T
                t1 = time.perf_counter()
                ids_d = ids.to(self.xfer.tgt_dev)
                new, cache = self._decode(model, cache, logits, ids_d, max_new)
                timing["decode_ms"] = (time.perf_counter() - t1) * 1000
            else:
                raise ValueError(f"unknown role {role!r}")
            timing["total_ms"] = (time.perf_counter() - t0) * 1000
        text = self.tok.decode(new, skip_special_tokens=True)
        st.history.append({"role": role, "tokens_in": T, "tokens_out": len(new)})
        return {"session": sid, "role": role, "text": text, "new_tokens": len(new), "timing": timing,
                "energy": em.reading.to_dict() if em.reading else None}

    def reset(self, sid: str) -> None:
        self.sessions.pop(sid, None)


def make_handler(esc: Escalator):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send(200, {"ok": True, "mapper": esc.xfer.mapper.summary(), "sessions": list(esc.sessions)})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/v1/generate":
                    self._send(200, esc.generate(req))
                elif self.path == "/v1/reset":
                    esc.reset(str(req.get("session", "default")))
                    self._send(200, {"ok": True})
                else:
                    self._send(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": str(e)})

        def log_message(self, *a):  # quiet
            pass

    return Handler


def serve(esc: Escalator, host: str = "127.0.0.1", port: int = 8765) -> HTTPServer:
    srv = HTTPServer((host, port), make_handler(esc))
    return srv
