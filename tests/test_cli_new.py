"""CLI coverage for doctor / pairs / plan / discover / experiment / inspect on tiny local models."""
import json

import pytest

from kvtransfer.cli import main

from test_cli import saved_pair  # noqa: F401  (fixture: tiny src/tgt dirs + tokenizer + calib.txt)


def test_doctor(capsys):
    main(["doctor"])
    out = capsys.readouterr().out
    assert "profile" in out and "memory pool" in out


def test_pairs_from_tags_and_ollama_config(tmp_path, capsys):
    main(["pairs", "--tags", "qwen2.5:7b-instruct,qwen2.5:14b-instruct,qwen3:4b-instruct", "--hardware", "dgx-spark",
          "--json", str(tmp_path / "pairs.json")])
    out = capsys.readouterr().out
    assert "mismatched-kv" in out and "cross-family" in out and "Qwen3-8B" in out
    j = json.loads((tmp_path / "pairs.json").read_text())
    assert any(p["category"] == "mismatched-kv" for p in j["pairs"])
    cfg = tmp_path / "config.toml"
    cfg.write_text('[ollama]\nllm = "qwen2.5:14b-instruct"\nragqa = "qwen2.5:7b-instruct"\njudge = "gemma4:26b"\n'
                   'postprocessjudges = ["qwen3:4b-instruct","Llama3.1:8b"]\n')
    main(["pairs", "--ollama-config", str(cfg), "--hardware", "dgx-spark"])
    out = capsys.readouterr().out
    assert "Qwen2.5-14B-Instruct" in out and "gemma-4-26b-it" in out and "Llama-3.1-8B-Instruct" in out


def test_plan_and_discover_on_saved_models(saved_pair, capsys):  # noqa: F811
    paths, data, root = saved_pair
    main(["plan", "--source", paths["src"], "--target", paths["tgt"], "--hardware", "dgx-spark"])
    out = capsys.readouterr().out
    assert "FITS" in out and "mappers" in out
    main(["discover", "--models-dir", str(root), "--no-hf-cache"])
    out = capsys.readouterr().out
    assert "src" in out and "tgt" in out and "small->large" in out


def test_experiment_cli_minimal(saved_pair, capsys):  # noqa: F811
    paths, data, root = saved_pair
    out_dir = root / "exp_cli"
    main(["experiment", "--source", paths["src"], "--target", paths["tgt"], "--device", "cpu", "--dtype", "float32",
          "--data", data, "--n-seqs", "8", "--seq-len", "24", "--stride", "2", "--batch-size", "4", "--k", "1,all",
          "--eval-n-seqs", "2", "--eval-seq-len", "24", "--eval-suffix-len", "4", "--bench-seq-lens", "8",
          "--bench-warmup", "1", "--bench-trials", "1", "--turns", "3", "--turn-tokens", "8", "--hardware", "dgx-spark",
          "--force", "--out", str(out_dir)])
    assert (out_dir / "report.md").exists() and (out_dir / "bench.json").exists()
    rep = json.loads((out_dir / "report.json").read_text())
    assert rep["best_k"] in (1, 3)
    main(["inspect", "--mapper", str(out_dir / "mappers" / f"k{rep['best_k']}")])
    assert "selected source layers" in capsys.readouterr().out
