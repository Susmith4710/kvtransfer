"""lm-evaluation-harness adapter: run downstream tasks through the transfer pipeline.

The paper (Sec. 4.2) reports transfer quality as downstream accuracy under lm-evaluation-harness
(ARC-Challenge, HellaSwag, WinoGrande, MMLU 5-shot, GSM8K 8-shot CoT) and summarises it as
``retention = transfer accuracy / target standalone accuracy``, plus a floor-normalised variant
that subtracts the chance level of each task.

This module registers an lm-eval model named ``"kvtransfer"``::

    lm_eval --model kvtransfer \\
        --model_args pretrained=Qwen/Qwen3-1.7B,source=Qwen/Qwen3-0.6B,mapper=mappers/qwen3-0.6b-to-1.7b \\
        --tasks arc_challenge,hellaswag --batch_size 1

Protocol per request: the *source* prefills all but the last ``hold_back`` context tokens, the
mapper converts that prefix into a target cache, and the *target* consumes the held-back context
tokens plus the continuation on top of the mapped cache.  Log-likelihoods are read from the
target's logits exactly as the stock ``hf`` model does.  Requests whose context is shorter than
``hold_back + 1`` tokens fall back to the target alone (there would be nothing to map).

``lm_eval`` is an optional dependency: this module imports it at top level but is itself never
imported by ``kvtransfer/__init__``, and the pure helpers (:func:`score_continuation`,
:func:`retention_table`) work without it.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .hf import forward_with_cache, load_model, prefill
from .mapper import Mapper
from .transfer import CrossModelTransfer, TransferResult, _pick

try:  # optional dependency
    from lm_eval.api.registry import register_model
    from lm_eval.models.huggingface import HFLM
    from lm_eval.models.utils import handle_stop_sequences, normalize_gen_kwargs, postprocess_generated_text

    _LM_EVAL_IMPORT_ERROR: Exception | None = None
except ImportError as _e:  # pragma: no cover - exercised only when lm_eval is missing
    _LM_EVAL_IMPORT_ERROR = _e
    HFLM = object  # type: ignore[assignment,misc]

    def register_model(*names):  # type: ignore[misc]
        return lambda cls: cls


__all__ = ["TransferLM", "score_continuation", "generate_tokens", "retention_table", "run_harness", "CHANCE"]

# Paper's chance floors for the floor-normalised retention (Sec. 4.2).  Task names are matched by
# exact key first, then by longest prefix, so ``mmlu_abstract_algebra`` -> ``mmlu``.
CHANCE: dict[str, float] = {
    "arc_challenge": 0.25,
    "hellaswag": 0.25,
    "winogrande": 0.5,
    "mmlu": 0.25,
    "gsm8k": 0.0,
}

# Metric keys tried, in order, when an lm-eval per-task result dict is given without a metric.
_METRIC_PREFERENCE = (
    "acc_norm,none", "acc,none", "exact_match,strict-match", "exact_match,flexible-extract", "exact_match,none",
    "acc_norm", "acc", "exact_match",
)


def _require_lm_eval() -> None:
    if _LM_EVAL_IMPORT_ERROR is not None:
        raise ImportError(
            "lm-evaluation-harness is required for kvtransfer.lm_eval_adapter.TransferLM / run_harness; "
            "install it with `pip install lm_eval`"
        ) from _LM_EVAL_IMPORT_ERROR


# ------------------------------------------------------------------------------ scoring protocol

@torch.no_grad()
def score_continuation(xfer: CrossModelTransfer, context_enc: Sequence[int], continuation_enc: Sequence[int],
                       hold_back: int, device, max_length: int | None = None) -> tuple[float, bool]:
    """Log-likelihood of ``continuation_enc`` given ``context_enc`` under the transfer protocol.

    Returns ``(sum of continuation token log-probs, greedy argmax reproduces the continuation)``,
    with the same semantics as lm-eval's ``HFLM._loglikelihood_tokens``: the model sees
    ``context + continuation[:-1]`` and the logits at the last ``len(continuation)`` positions score
    the continuation tokens.  If ``max_length`` is given the full sequence is left-truncated to
    ``max_length + 1`` tokens first (the split stays correct because the continuation is always the
    tail).  Contexts shorter than ``hold_back + 1`` tokens are scored on the target alone.
    """
    context_enc, continuation_enc = list(context_enc), list(continuation_enc)
    if not context_enc or not continuation_enc:
        raise ValueError("context and continuation must both be non-empty")
    seq = context_enc + continuation_enc
    if max_length is not None:
        seq = seq[-(max_length + 1):]
    contlen = len(continuation_enc)
    inp = seq[:-1]                       # last token is never fed: nothing predicts past it
    n_ctx = len(seq) - contlen           # context tokens surviving truncation
    logits = _protocol_logits(xfer, inp, n_ctx, hold_back, device)  # [len(inp), V]
    cont_logits = logits[-contlen:]
    return _score_from_logits(cont_logits, continuation_enc, device)


def _protocol_logits(xfer: CrossModelTransfer, inp: list[int], n_ctx: int, hold_back: int, device) -> torch.Tensor:
    """Logits [len(inp), V] for ``inp`` where the first ``n_ctx`` tokens are context, via the protocol."""
    hold_back = int(hold_back)
    if hold_back < 1:
        raise ValueError("hold_back must be >= 1")
    if n_ctx >= hold_back + 1:
        n_mapped = n_ctx - hold_back
        prefix = torch.tensor([inp[:n_mapped]], dtype=torch.long, device=device)
        tail = torch.tensor([inp[n_mapped:]], dtype=torch.long, device=device)
        _, src_cache = xfer.source_prefill(prefix)
        tgt_cache = xfer.map_cache(src_cache, n_mapped)
        out = forward_with_cache(xfer.target, tgt_cache, tail.to(xfer.tgt_dev), past_len=n_mapped)
        return out.logits[0]
    ids = torch.tensor([inp], dtype=torch.long, device=xfer.tgt_dev)
    return xfer.target(input_ids=ids).logits[0]


def _score_from_logits(cont_logits: torch.Tensor, continuation_enc: list[int], device) -> tuple[float, bool]:
    logprobs = F.log_softmax(cont_logits.float(), dim=-1)
    cont = torch.tensor(continuation_enc, dtype=torch.long, device=logprobs.device)
    greedy = bool(torch.equal(logprobs.argmax(-1), cont))
    ll = float(logprobs.gather(1, cont[:, None]).sum())
    return ll, greedy


# --------------------------------------------------------------------------------- generation

@torch.no_grad()
def generate_tokens(xfer: CrossModelTransfer, input_ids: torch.Tensor, max_new_tokens: int, hold_back: int,
                    eos_token_id: int | None = None, do_sample: bool = False,
                    temperature: float = 1.0) -> TransferResult:
    """``CrossModelTransfer.generate`` with the same short-context fallback as :func:`score_continuation`."""
    T = input_ids.shape[1]
    if T >= hold_back + 1:
        return xfer.generate(input_ids, max_new_tokens=max_new_tokens, hold_back=hold_back,
                             eos_token_id=eos_token_id, do_sample=do_sample, temperature=temperature)
    # target-only greedy/sampled decode
    ids = input_ids.to(xfer.tgt_dev)
    logits, cache = prefill(xfer.target, ids)
    seq = ids
    for _ in range(max_new_tokens):
        nxt = _pick(logits[:, -1], do_sample, temperature)
        seq = torch.cat([seq, nxt[:, None]], dim=1)
        if eos_token_id is not None and bool((nxt == eos_token_id).all()):
            break
        out = forward_with_cache(xfer.target, cache, nxt[:, None], past_len=seq.shape[1] - 1)
        logits, cache = out.logits, out.past_key_values
    return TransferResult(seq, T, 0, cache)


# ------------------------------------------------------------------------------- lm-eval model

@register_model("kvtransfer")
class TransferLM(HFLM):
    """lm-eval model: the target (``pretrained``) answers from a cache the ``source`` prefilled.

    Extra ``model_args`` on top of the stock ``hf`` model: ``source`` (model id/path), ``mapper``
    (directory written by :meth:`Mapper.save`) and ``hold_back`` (default 1).  Everything else
    (``dtype``, ``device``, ``batch_size``, ``max_length``, ...) is HFLM's.  Requests are processed
    one at a time regardless of ``batch_size``.
    """

    def __init__(self, source: str, pretrained: str, mapper: str, hold_back: int = 1, **kwargs) -> None:
        _require_lm_eval()
        super().__init__(pretrained=pretrained, **kwargs)
        if self.backend != "causal":
            raise ValueError("kvtransfer only supports decoder-only (causal) targets")
        tgt_dtype = next(self.model.parameters()).dtype
        self.source_model = load_model(source, device=self.device, dtype=tgt_dtype)
        self.mapper = Mapper.load(mapper)
        self.xfer = CrossModelTransfer(self.source_model, self.model, self.mapper)
        self.hold_back = int(hold_back)
        if self.hold_back < 1:
            raise ValueError("hold_back must be >= 1")
        self.source_name = source

    # ---- scoring ------------------------------------------------------------------------------
    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False, override_bs: int | None = None):
        from tqdm import tqdm

        res = []
        pbar = tqdm(total=len(requests), disable=(disable_tqdm or (self.rank != 0)),
                    desc="Running loglikelihood requests (kvtransfer)")
        for request_str, context_enc, continuation_enc in requests:
            assert len(context_enc) > 0 and len(continuation_enc) > 0
            assert len(continuation_enc) <= self.max_length
            answer = score_continuation(self.xfer, context_enc, continuation_enc, self.hold_back, self.device,
                                        max_length=self.max_length)
            res.append(answer)
            if request_str is not None:  # loglikelihood_rolling passes None and caches per example itself
                self.cache_hook.add_partial("loglikelihood", request_str, answer)
            pbar.update(1)
        pbar.close()
        return res

    # ---- generation ---------------------------------------------------------------------------
    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        from tqdm import tqdm

        res = []
        eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
        pbar = tqdm(total=len(requests), disable=(disable_tqdm or (self.rank != 0)),
                    desc="Running generate_until requests (kvtransfer)")
        for req in requests:
            context, gen_kwargs = req.args
            if not isinstance(gen_kwargs, dict):
                raise TypeError(f"expected gen_kwargs dict, got {type(gen_kwargs)}")
            kwargs = normalize_gen_kwargs(gen_kwargs, self.max_gen_toks)
            until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
            max_gen_toks = int(kwargs.pop("max_gen_toks"))
            do_sample = bool(kwargs.pop("do_sample", False))
            temperature = kwargs.pop("temperature", 1.0)
            temperature = float(temperature) if temperature else 1.0
            max_ctx_len = self.max_length - max_gen_toks
            if max_ctx_len <= 0:
                raise ValueError(f"max_gen_toks ({max_gen_toks}) must be smaller than max_length ({self.max_length})")
            context_enc = self.tok_encode(context, left_truncate_len=max_ctx_len) or [self.prefix_token_id]
            ids = torch.tensor([context_enc], dtype=torch.long, device=self.device)
            out = generate_tokens(self.xfer, ids, max_gen_toks, self.hold_back, eos_token_id=self.eot_token_id,
                                  do_sample=do_sample, temperature=temperature)
            cont_toks = out.tokens[0, out.n_prompt:].tolist()
            if isinstance(self.think_end_token, int):
                idx = [i for i, t in enumerate(cont_toks) if t == self.think_end_token]
                if idx:
                    cont_toks = cont_toks[idx[-1] + 1:]
            s = self.tok_decode(cont_toks)
            if isinstance(self.think_end_token, int):
                s = s.lstrip()
            s = postprocess_generated_text(generation=s, stop=until,
                                           think_end_token=self.think_end_token
                                           if isinstance(self.think_end_token, str) else None)
            res.append(s)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), s)
            pbar.update(1)
        pbar.close()
        return res


# --------------------------------------------------------------------------------- retention

def _chance_for(task: str, chance: dict[str, float] | None) -> float:
    table = CHANCE if chance is None else chance
    if task in table:
        return float(table[task])
    best = [k for k in table if task.startswith(k)]
    if best:
        return float(table[max(best, key=len)])
    return 0.0


def _pick_metric(entry: Any, metric: str | None) -> tuple[str | None, float | None]:
    """``(metric_key, value)`` from a float or an lm-eval per-task result dict."""
    if entry is None:
        return None, None
    if isinstance(entry, (int, float)):
        return metric, float(entry)
    if not isinstance(entry, dict):
        raise TypeError(f"unsupported result entry {type(entry)}")
    if metric is not None:
        if metric in entry:
            return metric, float(entry[metric])
        for key in entry:
            if key.split(",")[0] == metric and "stderr" not in key:
                return key, float(entry[key])
        return metric, None
    for key in _METRIC_PREFERENCE:
        if key in entry and isinstance(entry[key], (int, float)):
            return key, float(entry[key])
    for key, val in entry.items():
        if isinstance(val, (int, float)) and not isinstance(val, bool) and "stderr" not in key and key != "alias":
            return key, float(val)
    return None, None


def retention_table(results_transfer: dict, results_target: dict, results_source: dict | None = None,
                    chance: dict[str, float] | None = None, metric: str | None = None) -> list[dict]:
    """Per-task retention of the transfer run relative to the target run (paper Table 2/3).

    Each ``results_*`` is ``{task: value}`` or lm-eval's ``results["results"]`` (``{task: {metric_key:
    value, ...}}``); ``metric`` picks the key (``"acc_norm,none"``, or ``"acc_norm"`` to match any
    filter), otherwise a sensible default is chosen per task.  Rows carry ``retention_pct`` =
    100 * transfer / target and ``normalized_retention_pct`` = 100 * (transfer - c) / (target - c) with
    the chance floor ``c`` from ``chance`` (default :data:`CHANCE`; unknown tasks use 0).
    """
    rows = []
    for task in results_transfer:
        if task not in results_target:
            continue
        key, transfer = _pick_metric(results_transfer[task], metric)
        _, target = _pick_metric(results_target[task], key if key is not None else metric)
        _, source = _pick_metric((results_source or {}).get(task), key if key is not None else metric)
        c = _chance_for(task, chance)
        retention = norm = None
        if transfer is not None and target is not None:
            retention = 100.0 * transfer / target if target != 0 else None
            denom = target - c
            norm = 100.0 * (transfer - c) / denom if not math.isclose(denom, 0.0) else None
        rows.append({
            "task": task, "metric": key, "chance": c, "source": source, "target": target, "transfer": transfer,
            "retention_pct": retention, "normalized_retention_pct": norm,
        })
    return rows


# ------------------------------------------------------------------------------- convenience

def run_harness(source: str, target: str, mapper: str, tasks: Sequence[str] | str, device: str = "cuda",
                dtype: str = "auto", limit: int | float | None = None, hold_back: int = 1,
                num_fewshot: int | None = None, batch_size: int | str = 1, metric: str | None = None,
                **simple_evaluate_kwargs) -> dict:
    """Run ``lm_eval.simple_evaluate`` for target-only, source-only and transfer; return all three plus
    :func:`retention_table` rows under ``"retention"``."""
    _require_lm_eval()
    import lm_eval

    if isinstance(tasks, str):
        tasks = [t.strip() for t in tasks.split(",") if t.strip()]
    tasks = list(tasks)
    common = dict(tasks=tasks, num_fewshot=num_fewshot, batch_size=batch_size, device=device, limit=limit,
                  **simple_evaluate_kwargs)
    res_target = lm_eval.simple_evaluate(model="hf", model_args={"pretrained": target, "dtype": dtype}, **common)
    res_source = lm_eval.simple_evaluate(model="hf", model_args={"pretrained": source, "dtype": dtype}, **common)
    res_xfer = lm_eval.simple_evaluate(
        model="kvtransfer",
        model_args={"pretrained": target, "source": source, "mapper": mapper, "dtype": dtype, "hold_back": hold_back},
        **common,
    )
    table = retention_table(res_xfer["results"], res_target["results"], res_source["results"], metric=metric)
    return {"target": res_target, "source": res_source, "transfer": res_xfer, "retention": table}
