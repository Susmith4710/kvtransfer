"""kvtransfer: cross-model KV cache transfer within an LLM family.

Implements the closed-form per-head ridge mapper of Heo et al., "Cross-Model KV Cache Transfer in
LLM Families: A Closed-Form Linear Mapping for Prefill Reuse" (arXiv:2608.03893) on top of Hugging
Face transformers models.

Typical use::

    from kvtransfer import calibrate, Mapper, CrossModelTransfer, load_model, load_tokenizer
    src, tgt = load_model("Qwen/Qwen3-0.6B"), load_model("Qwen/Qwen3-1.7B")
    stats = calibrate(src, tgt, token_batches, stride=4)      # one streaming pass, both models
    mapper = Mapper.fit(stats, k=8, lam=0.01)                  # closed-form, any k from the same stats
    mapper.save("mappers/qwen3-0.6b-to-1.7b")
    xfer = CrossModelTransfer(src, tgt, mapper)
    out = xfer.generate(input_ids, max_new_tokens=64)          # source prefill -> mapped cache -> target decode
"""
from .bench import benchmark, format_rows
from .calibration import CalibrationStats, calibrate
from .hf import ModelSpec, check_matched_kv, load_model, load_tokenizer, model_spec
from .mapper import Mapper
from .metrics import EvalReport, evaluate
from .rope import RopeCodec
from .select import probe_r2, selection_score, top_k_layers
from .transfer import CrossModelTransfer, Session, TransferResult
from .hf import load_pair, encode_prompt, tokenizer_compatibility
from .hardware import HardwareProfile, PairPlan, DGX_SPARK, detect as detect_hardware
from .experiment import ExperimentConfig, run_experiment
from .energy import EnergyMeter

__version__ = "0.1.0"
__all__ = [
    "CalibrationStats", "calibrate", "Mapper", "CrossModelTransfer", "Session", "TransferResult",
    "ModelSpec", "model_spec", "check_matched_kv", "load_model", "load_tokenizer", "RopeCodec",
    "probe_r2", "selection_score", "top_k_layers", "evaluate", "EvalReport", "benchmark", "format_rows",
    "load_pair", "encode_prompt", "tokenizer_compatibility", "HardwareProfile", "PairPlan", "DGX_SPARK",
    "detect_hardware", "ExperimentConfig", "run_experiment", "EnergyMeter",
]
