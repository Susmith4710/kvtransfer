"""lm-evaluation-harness adapter, offline on CPU with tiny random models."""
import math

import pytest
import torch
import torch.nn.functional as F

from kvtransfer import CrossModelTransfer, Mapper, calibrate
from kvtransfer.lm_eval_adapter import CHANCE, generate_tokens, retention_table, score_continuation

from conftest import VOCAB, random_batches, tiny_qwen3

lm_eval = pytest.importorskip("lm_eval")

SENTENCES = [
    "the quick brown fox jumps over the lazy dog", "a small model prefills and a large model decodes",
    "keys are rotated by position and values are not", "ridge regression is a closed form linear solve",
    "calibration uses a few hundred sequences of prose", "model families share a tokenizer across sizes",
] * 20


def _identity_mapper(model):
    stats = calibrate(model, model, random_batches(n_batches=8, batch=4, T=64, seed=7), stride=1)
    return Mapper.fit(stats, k=1, lam=1e-6)


@pytest.fixture(scope="module")
def identity_xfer():
    model = tiny_qwen3(3, seed=1)
    return CrossModelTransfer(model, model, _identity_mapper(model))


@torch.no_grad()
def _plain_score(model, context_enc, continuation_enc):
    """Single-model reference: sum of log-softmax at the continuation positions, HFLM semantics."""
    seq = context_enc + continuation_enc
    logits = model(input_ids=torch.tensor([seq[:-1]])).logits[0].float()
    lp = F.log_softmax(logits, dim=-1)[-len(continuation_enc):]
    cont = torch.tensor(continuation_enc)
    return float(lp.gather(1, cont[:, None]).sum()), bool(torch.equal(lp.argmax(-1), cont))


def _ids(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (n,), generator=g).tolist()


# ------------------------------------------------------------------------- score_continuation

@pytest.mark.parametrize("hold_back", [1, 3])
def test_score_continuation_identity_matches_plain(identity_xfer, hold_back):
    ctx, cont = _ids(30, seed=3), _ids(6, seed=4)
    ll_ref, greedy_ref = _plain_score(identity_xfer.target, ctx, cont)
    ll, greedy = score_continuation(identity_xfer, ctx, cont, hold_back=hold_back, device="cpu")
    assert isinstance(ll, float) and isinstance(greedy, bool)
    assert abs(ll - ll_ref) < 2e-2
    assert greedy == greedy_ref


def test_score_continuation_single_token_continuation(identity_xfer):
    ctx, cont = _ids(12, seed=5), _ids(1, seed=6)
    ll_ref, greedy_ref = _plain_score(identity_xfer.target, ctx, cont)
    ll, greedy = score_continuation(identity_xfer, ctx, cont, hold_back=1, device="cpu")
    assert abs(ll - ll_ref) < 2e-2 and greedy == greedy_ref


def test_score_continuation_short_context_falls_back_to_target(identity_xfer, monkeypatch):
    ctx, cont = _ids(2, seed=8), _ids(5, seed=9)  # 2 context tokens < hold_back + 1 = 4
    monkeypatch.setattr(identity_xfer, "source_prefill", lambda *a, **k: pytest.fail("source must not run"))
    ll_ref, greedy_ref = _plain_score(identity_xfer.target, ctx, cont)
    ll, greedy = score_continuation(identity_xfer, ctx, cont, hold_back=3, device="cpu")
    assert ll == pytest.approx(ll_ref, abs=1e-6) and greedy == greedy_ref


def test_score_continuation_left_truncates_to_max_length(identity_xfer):
    ctx, cont = _ids(40, seed=10), _ids(4, seed=11)
    max_length = 20
    kept = (ctx + cont)[-(max_length + 1):]
    ll_ref, greedy_ref = _plain_score(identity_xfer.target, kept[:-len(cont)], cont)
    ll, greedy = score_continuation(identity_xfer, ctx, cont, hold_back=1, device="cpu", max_length=max_length)
    assert abs(ll - ll_ref) < 2e-2 and greedy == greedy_ref


def test_score_continuation_rejects_bad_inputs(identity_xfer):
    with pytest.raises(ValueError):
        score_continuation(identity_xfer, [], [1], hold_back=1, device="cpu")
    with pytest.raises(ValueError):
        score_continuation(identity_xfer, [1, 2, 3], [4], hold_back=0, device="cpu")


def test_generate_tokens_fallback_and_transfer(identity_xfer):
    ids = torch.tensor([_ids(10, seed=12)])
    out = generate_tokens(identity_xfer, ids, max_new_tokens=3, hold_back=1)
    assert out.tokens.shape == (1, 13) and out.n_prompt == 10 and out.n_mapped == 9
    short = torch.tensor([_ids(1, seed=13)])
    out = generate_tokens(identity_xfer, short, max_new_tokens=3, hold_back=1)
    assert out.tokens.shape == (1, 4) and out.n_mapped == 0
    # greedy target-only decode is what the fallback must reproduce
    ref = identity_xfer.target.generate(short, max_new_tokens=3, do_sample=False)
    assert torch.equal(out.tokens, ref)


# ----------------------------------------------------------------------------- retention_table

def test_retention_table_numbers():
    rows = retention_table(
        {"arc_challenge": 0.75, "gsm8k": 0.40, "mmlu_abstract_algebra": {"acc,none": 0.5, "acc_stderr,none": 0.01},
         "unknown_task": {"f1,none": 0.5}},
        {"arc_challenge": 0.80, "gsm8k": 0.50, "mmlu_abstract_algebra": {"acc,none": 0.75, "acc_stderr,none": 0.01},
         "unknown_task": {"f1,none": 1.0}},
        {"arc_challenge": 0.60},
    )
    by = {r["task"]: r for r in rows}
    arc = by["arc_challenge"]
    assert arc["chance"] == 0.25 and arc["source"] == 0.60
    assert arc["retention_pct"] == pytest.approx(93.75)
    assert arc["normalized_retention_pct"] == pytest.approx(100 * 0.5 / 0.55)  # 90.909...
    gsm = by["gsm8k"]
    assert gsm["chance"] == 0.0 and gsm["retention_pct"] == pytest.approx(80.0)
    assert gsm["normalized_retention_pct"] == pytest.approx(gsm["retention_pct"])
    mm = by["mmlu_abstract_algebra"]  # prefix match on the paper's chance table, metric picked from the dict
    assert mm["chance"] == 0.25 and mm["metric"] == "acc,none"
    assert mm["retention_pct"] == pytest.approx(100 * 0.5 / 0.75)
    assert mm["normalized_retention_pct"] == pytest.approx(100 * 0.25 / 0.5)
    unk = by["unknown_task"]
    assert unk["chance"] == 0.0 and unk["retention_pct"] == pytest.approx(50.0)
    assert CHANCE["winogrande"] == 0.5


def test_retention_table_metric_override_and_degenerate_denominator():
    xfer = {"hellaswag": {"acc,none": 0.30, "acc_norm,none": 0.40}}
    tgt = {"hellaswag": {"acc,none": 0.30, "acc_norm,none": 0.25}}
    (row,) = retention_table(xfer, tgt, metric="acc")
    assert row["metric"] == "acc,none" and row["retention_pct"] == pytest.approx(100.0)
    (row,) = retention_table(xfer, tgt, metric="acc_norm,none")
    assert row["retention_pct"] == pytest.approx(160.0) and row["normalized_retention_pct"] is None  # target == chance
    assert retention_table({"a": 0.5}, {"b": 0.5}) == []


# ------------------------------------------------------------------------ TransferLM end to end

@pytest.fixture(scope="module")
def saved_identity(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    root = tmp_path_factory.mktemp("lm_eval_models")
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(SENTENCES, trainers.BpeTrainer(vocab_size=257, special_tokens=["<unk>", "<eos>"]))
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", eos_token="<eos>")
    model = tiny_qwen3(3, seed=1)
    model_dir = root / "model"
    model.save_pretrained(model_dir)
    hf_tok.save_pretrained(model_dir)
    mapper_dir = root / "mapper"
    _identity_mapper(model).save(mapper_dir)
    return str(model_dir), str(mapper_dir)


@pytest.fixture(scope="module")
def transfer_lm(saved_identity):
    from kvtransfer.lm_eval_adapter import TransferLM

    model_dir, mapper_dir = saved_identity
    return TransferLM(source=model_dir, pretrained=model_dir, mapper=mapper_dir, hold_back=1,
                      device="cpu", dtype="float32", batch_size=1)


@pytest.fixture(scope="module")
def stock_lm(saved_identity):
    from lm_eval.models.huggingface import HFLM

    model_dir, _ = saved_identity
    return HFLM(pretrained=model_dir, device="cpu", dtype="float32", batch_size=1)


def _ll_instances():
    from lm_eval.api.instance import Instance

    pairs = [("the quick brown fox jumps over", " the lazy dog"),
             ("keys are rotated by position", " and values are not"),
             ("a small model", " prefills")]
    return [Instance(request_type="loglikelihood", doc={}, arguments=p, idx=i) for i, p in enumerate(pairs)]


def test_transfer_lm_is_registered():
    from lm_eval.api.registry import get_model
    from kvtransfer.lm_eval_adapter import TransferLM

    assert get_model("kvtransfer") is TransferLM


def test_transfer_lm_construction(transfer_lm, saved_identity):
    model_dir, _ = saved_identity
    assert transfer_lm.hold_back == 1 and transfer_lm.backend == "causal"
    assert transfer_lm.xfer.target is transfer_lm.model
    assert next(transfer_lm.source_model.parameters()).dtype == torch.float32
    assert transfer_lm.mapper.k == 1


def test_transfer_lm_loglikelihood_matches_stock_hflm(transfer_lm, stock_lm):
    reqs = _ll_instances()
    got = transfer_lm.loglikelihood(reqs, disable_tqdm=True)
    ref = stock_lm.loglikelihood(_ll_instances(), disable_tqdm=True)
    assert len(got) == len(ref) == 3
    for (ll, greedy), (ll_ref, greedy_ref) in zip(got, ref):
        assert isinstance(ll, float) and isinstance(greedy, bool)
        assert abs(ll - ll_ref) < 2e-2
        assert greedy == greedy_ref


def test_transfer_lm_loglikelihood_empty_context_uses_prefix_token(transfer_lm, stock_lm):
    from lm_eval.api.instance import Instance

    req = lambda: [Instance(request_type="loglikelihood", doc={}, arguments=("", "the quick brown fox"), idx=0)]
    (ll, greedy), = transfer_lm.loglikelihood(req(), disable_tqdm=True)
    (ll_ref, greedy_ref), = stock_lm.loglikelihood(req(), disable_tqdm=True)
    assert abs(ll - ll_ref) < 2e-2 and greedy == greedy_ref


def test_transfer_lm_loglikelihood_rolling_runs(transfer_lm):
    from lm_eval.api.instance import Instance

    reqs = [Instance(request_type="loglikelihood_rolling", doc={}, arguments=("the quick brown fox jumps",), idx=0)]
    (ll,) = transfer_lm.loglikelihood_rolling(reqs, disable_tqdm=True)
    assert isinstance(ll, float) and math.isfinite(ll)


def test_transfer_lm_generate_until(transfer_lm, stock_lm):
    from lm_eval.api.instance import Instance

    reqs = [Instance(request_type="generate_until", doc={},
                     arguments=("the quick brown fox", {"until": ["\n"], "max_gen_toks": 4}), idx=0),
            Instance(request_type="generate_until", doc={},
                     arguments=("keys are rotated", {"until": ["\n", "dog"], "max_gen_toks": 2}), idx=1)]
    out = transfer_lm.generate_until(reqs, disable_tqdm=True)
    assert isinstance(out, list) and len(out) == 2 and all(isinstance(s, str) for s in out)
    assert "\n" not in out[0] and "dog" not in out[1]
    # identity pair: greedy transfer decode agrees with the stock model's greedy decode
    ref = stock_lm.generate_until(reqs, disable_tqdm=True)
    assert out == ref


def test_transfer_lm_cache_hook_receives_partials(transfer_lm):
    seen = []

    class Hook:
        def add_partial(self, attr, req, res):
            seen.append((attr, req, res))

    old = transfer_lm.cache_hook
    transfer_lm.set_cache_hook(Hook())
    try:
        transfer_lm.loglikelihood(_ll_instances()[:1], disable_tqdm=True)
        from lm_eval.api.instance import Instance
        transfer_lm.generate_until([Instance(request_type="generate_until", doc={},
                                             arguments=("the quick", {"until": ["\n"], "max_gen_toks": 2}), idx=0)],
                                   disable_tqdm=True)
    finally:
        transfer_lm.set_cache_hook(old)
    assert [s[0] for s in seen] == ["loglikelihood", "generate_until"]
    assert seen[0][1] == ("the quick brown fox jumps over", " the lazy dog")
    assert seen[1][1] == ("the quick", {"until": ["\n"], "max_gen_toks": 2})


def test_run_harness_requires_lm_eval_message(monkeypatch):
    import kvtransfer.lm_eval_adapter as mod

    monkeypatch.setattr(mod, "_LM_EVAL_IMPORT_ERROR", ImportError("nope"))
    with pytest.raises(ImportError, match="lm-evaluation-harness is required"):
        mod.run_harness("s", "t", "m", ["arc_challenge"])
