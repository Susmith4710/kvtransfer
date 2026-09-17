"""CLI end to end on tiny models saved to disk with a tiny BPE tokenizer trained locally (offline)."""
import json

import pytest
import torch

from kvtransfer.cli import main

from conftest import tiny_qwen3

SENTENCES = [
    "the quick brown fox jumps over the lazy dog", "a small model prefills and a large model decodes",
    "keys are rotated by position and values are not", "ridge regression is a closed form linear solve",
    "calibration uses a few hundred sequences of prose", "model families share a tokenizer across sizes",
] * 20


@pytest.fixture(scope="module")
def saved_pair(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    root = tmp_path_factory.mktemp("models")
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(SENTENCES, trainers.BpeTrainer(vocab_size=257, special_tokens=["<unk>", "<eos>"]))
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", eos_token="<eos>")
    paths = {}
    for name, m in (("src", tiny_qwen3(3, seed=1)), ("tgt", tiny_qwen3(4, seed=2))):
        d = root / name
        m.save_pretrained(d)
        hf_tok.save_pretrained(d)
        paths[name] = str(d)
    data = root / "calib.txt"
    data.write_text("\n\n".join(" ".join(SENTENCES[i:i + 12]) for i in range(0, len(SENTENCES) - 12, 3)))
    return paths, str(data), root


def test_check(saved_pair, capsys):
    paths, _, _ = saved_pair
    main(["check", "--source", paths["src"], "--target", paths["tgt"], "--k", "1,2"])
    out = capsys.readouterr().out
    assert "matched-KV: yes" in out and "shared tokenizer: yes" in out and "k=2:" in out


def test_fit_inspect_eval_bench_generate(saved_pair, capsys):
    paths, data, root = saved_pair
    out_dir = root / "mappers"
    stats_dir = root / "stats"
    common = ["--source", paths["src"], "--target", paths["tgt"], "--device", "cpu"]
    main(["fit", *common, "--data", data, "--n-seqs", "8", "--seq-len", "24", "--stride", "2", "--batch-size", "4",
          "--k", "1,all", "--out", str(out_dir), "--stats", str(stats_dir)])
    assert (out_dir / "k1" / "mapper.safetensors").exists() and (out_dir / "k3" / "mapper.json").exists()
    assert (stats_dir / "stats_K.safetensors").exists() and (out_dir / "selection_r2.json").exists()
    # refit from saved stats without loading models
    main(["fit", *common, "--data", data, "--k", "2", "--out", str(out_dir), "--stats", str(stats_dir)])
    assert json.loads((out_dir / "k2" / "mapper.json").read_text())["k"] == 2

    main(["inspect", "--mapper", str(out_dir / "k2")])
    assert "target   0 <-" in capsys.readouterr().out

    rep = root / "eval.json"
    main(["eval", *common, "--mapper", str(out_dir / "k2"), "--data", data, "--n-seqs", "3", "--seq-len", "24",
          "--suffix-len", "4", "--out", str(rep)])
    assert json.loads(rep.read_text())["n_sequences"] == 3

    main(["bench", *common, "--mapper", str(out_dir / "k2"), "--seq-lens", "8,16", "--warmup", "1", "--trials", "1"])
    assert "speedup" in capsys.readouterr().out

    main(["generate", *common, "--mapper", str(out_dir / "k2"), "--prompt", "the quick brown fox",
          "--max-new-tokens", "4", "--compare"])
    out = capsys.readouterr().out
    assert "target from mapped cache" in out and "source standalone" in out
