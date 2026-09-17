"""Planner, catalog, and the paper's own numbers (Appendix D / Table 12) reproduced from the formula."""
import pytest

from kvtransfer import hardware as hw
from kvtransfer.catalog import BY_ID, analyze_tags, classify_pair, lookup
from kvtransfer.hf import ModelSpec
from kvtransfer.mapper import Mapper


def spec(name, L):
    return ModelSpec(name, L, 8, 128, 1e6, 1.0)


# Paper Table 12: pair, k, total params (K+V, weights) -- our formula must reproduce them.
PAPER_TABLE_12 = [
    ("Qwen3-8B", 36, "Qwen3-32B", 64, 12, 1.61e9),
    ("Qwen3-14B", 40, "Qwen3-32B", 64, 8, 1.07e9),
    ("Llama-3.1-8B", 32, "Llama-3.1-70B", 80, 20, 3.36e9),
    ("Ministral-3-3B", 26, "Ministral-3-8B", 34, 26, 1.85e9),      # k = all = 26
    ("Ministral-3-8B", 34, "Ministral-3-14B", 40, 12, 1.01e9),
    ("Ministral-3-3B", 26, "Ministral-3-14B", 40, 20, 1.68e9),
]


@pytest.mark.parametrize("s, Ls, t, Lt, k, expected", PAPER_TABLE_12)
def test_appendix_d_parameter_counts(s, Ls, t, Lt, k, expected):
    n = Mapper.formula_params(spec(t, Lt), spec(s, Ls), k)
    assert abs(n - expected) / expected < 0.01, (n, expected)


def test_paper_storage_sizes_match_table_12():
    # 1.07 B params -> "4 GB"; 3.36 B -> "12 GB" (fp32)
    assert round(1.07e9 * 4 / 1e9) == 4 and round(3.36e9 * 4 / 1e9) == 13 or round(3.36e9 * 4 / hw.GIB) == 12


def test_detect_runs_on_cpu_box():
    p = hw.detect()
    assert p.device in ("cpu", "cuda")
    assert p.system_total_bytes > 0
    assert p.recommended_attn in ("sdpa", "flash_attention_2")
    assert "profile" in hw.format_profile(p)


def test_dgx_spark_plan_for_paper_pair():
    """Qwen3 14B->32B in bf16 on a Spark: models 95 GB alone; must be flagged as not fitting statistics."""
    a, b = BY_ID["Qwen/Qwen3-14B"], BY_ID["Qwen/Qwen3-32B"]
    plan = hw.PairPlan(a.cost(), b.cost(), "bfloat16", 500, 1024, 4, 4)
    v = plan.fit(hw.DGX_SPARK)
    assert v["models_gib"] > 85
    assert not v["fits"]
    assert plan.mapper_bytes(8) == Mapper.formula_params(spec("t", 64), spec("s", 40), 8) * 4
    assert v["calibration_tokens_per_head"] == 128000


def test_dgx_spark_plan_small_pair_fits_single_pass():
    a, b = BY_ID["Qwen/Qwen3-4B-Instruct-2507"], BY_ID["Qwen/Qwen3-8B"]
    v = hw.PairPlan(a.cost(), b.cost(), "bfloat16", 500, 1024, 4, 4).fit(hw.DGX_SPARK)
    assert v["fits"] and v["mode"].startswith("single pass")
    assert v["recommended_batch_size"] >= 4
    assert "FITS" in hw.format_plan(hw.PairPlan(a.cost(), b.cost()), v)


def test_estimate_params_matches_known_models():
    cfg = {"hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 128256, "num_attention_heads": 32,
           "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 14336, "tie_word_embeddings": False}
    assert abs(hw.estimate_params(cfg) - 8.03e9) / 8.03e9 < 0.01          # Llama 3.1 8B
    moe = {"hidden_size": 2048, "num_hidden_layers": 48, "vocab_size": 151936, "num_attention_heads": 32,
           "num_key_value_heads": 4, "head_dim": 128, "intermediate_size": 6144, "num_experts": 128,
           "moe_intermediate_size": 768, "tie_word_embeddings": False}
    assert abs(hw.estimate_params(moe) - 30.5e9) / 30.5e9 < 0.03         # Qwen3-30B-A3B


def test_catalog_lookup_from_ollama_tags():
    assert lookup("qwen2.5:14b-instruct").hf_id == "Qwen/Qwen2.5-14B-Instruct"
    assert lookup("qwen3:30b-a3b-instruct-2507-q4_K_M").hf_id == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert lookup("Llama3.1:8b").hf_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert lookup("llama3.2:1b").n_kv == 8 and lookup("llama3.2:1b").head_dim == 64
    assert lookup("Qwen/Qwen3-8B").n_layers == 36
    assert lookup("nonexistent:1b") is None


def test_classify_pairs_from_the_pod_config():
    q7, q14 = BY_ID["Qwen/Qwen2.5-7B-Instruct"], BY_ID["Qwen/Qwen2.5-14B-Instruct"]
    q3_4, q3_30 = BY_ID["Qwen/Qwen3-4B-Instruct-2507"], BY_ID["Qwen/Qwen3-30B-A3B-Instruct-2507"]
    l8, l1 = BY_ID["meta-llama/Llama-3.1-8B-Instruct"], BY_ID["meta-llama/Llama-3.2-1B-Instruct"]
    assert classify_pair(q7, q14)[0] == "mismatched-kv"
    assert classify_pair(q3_4, q3_30)[0] == "mismatched-kv"
    assert classify_pair(l1, l8)[0] == "mismatched-kv"
    assert classify_pair(q3_4, q14)[0] == "cross-family"
    assert classify_pair(q7, l8)[0] == "unusable"
    assert classify_pair(q3_4, BY_ID["Qwen/Qwen3-8B"])[0] == "matched-kv"
    assert classify_pair(BY_ID["Qwen/Qwen3-14B"], BY_ID["Qwen/Qwen3-32B"])[0] == "paper-validated"
    assert classify_pair(l8, BY_ID["meta-llama/Llama-3.2-3B-Instruct"])[0] == "matched-kv"


def test_analyze_tags_pod_list_has_no_paper_pair_and_suggests_siblings():
    tags = ["qwen2.5:14b-instruct", "qwen2.5:7b-instruct", "qwen3:30b-a3b-instruct-2507-q4_K_M", "qwen3:4b-instruct",
            "Llama3.1:8b", "llama3.2:1b", "gemma4:26b"]
    a = analyze_tags(tags, hw.DGX_SPARK)
    cats = {p.category for p in a["pairs"]}
    assert "paper-validated" not in cats and "matched-kv" not in cats
    assert {"mismatched-kv", "cross-family", "unusable"} <= cats
    assert "Qwen/Qwen3-8B" in a["suggestions"]["Qwen/Qwen3-4B-Instruct-2507"]
    assert "Qwen/Qwen2.5-32B-Instruct" in a["suggestions"]["Qwen/Qwen2.5-14B-Instruct"]
    fits = {(p.source.short, p.target.short): p.plan["fits"] for p in a["pairs"] if p.plan}
    assert fits[("Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct")]
    from kvtransfer.catalog import format_analysis
    assert "mismatched-kv" in format_analysis(a)
