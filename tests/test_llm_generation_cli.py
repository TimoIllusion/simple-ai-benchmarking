import argparse

from simple_ai_benchmarking.llm_generation import (
    build_generation_config_from_args,
    parse_arguments,
)
from simple_ai_benchmarking.workloads.llm_workload import (
    OLLAMA_BACKEND,
    PYTORCH_GENERATION_BACKEND,
)


def test_saib_llm_runs_all_local_workloads_with_no_args(monkeypatch):
    # Bare `saib-llm` should build all local LM workloads (simple transformer,
    # KV-cache decoder, Hugging Face), mirroring how `saib-pt` runs several models.
    import pytest

    pytest.importorskip("torch")
    from simple_ai_benchmarking.config_pt_tf import get_device_name_pytorch
    from simple_ai_benchmarking.llm_generation import build_generation_configs
    from simple_ai_benchmarking.workloads.llm_workload import (
        HF_CAUSAL_BACKEND,
        PYTORCH_KV_DECODER_BACKEND,
    )

    monkeypatch.setattr("sys.argv", ["saib-llm"])

    args = parse_arguments()
    assert args.backend is None
    configs = build_generation_configs(args)

    assert [c.backend for c in configs] == [
        PYTORCH_GENERATION_BACKEND,
        PYTORCH_KV_DECODER_BACKEND,
        HF_CAUSAL_BACKEND,
    ]
    # Reference transformer keeps its defaults (self-contained, no server).
    assert configs[0].model == "SimpleTransformerLM"
    assert configs[0].device_name == get_device_name_pytorch()
    assert configs[0].base_url == ""
    # HF backend gets a real repo id; heavier backends default to bf16.
    assert configs[2].model
    assert configs[1].compute_precision == "BF16"
    assert configs[2].compute_precision == "BF16"


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(
        backend=OLLAMA_BACKEND,
        base_url=None,
        model="llama3",
        requests=1,
        warmup_requests=0,
        concurrency=1,
        prompt_tokens=1,
        generated_tokens=1,
        context_length=2048,
        timeout_s=1.0,
        api_key=None,
        api_key_env="OPENAI_API_KEY",
        compute_precision="",
        quantization="",
        model_params=0,
        ai_framework_version="",
        ai_framework_extra_info="",
        accelerator="cpu",
        weight_source="",
        device="cpu",
        vocab_size=32000,
        embedding_dim=256,
        transformer_layers=4,
        attention_heads=4,
        feedforward_dim=1024,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_generation_config_uses_backend_default_base_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test")

    config = build_generation_config_from_args(_args(backend=OLLAMA_BACKEND))

    assert config.backend == OLLAMA_BACKEND
    assert config.base_url == "http://ollama.test"
    assert config.context_length == 2048
    assert config.model_cfg.context_length == 2048


def test_build_generation_config_pytorch_metadata_defaults():
    import pytest

    pytest.importorskip("torch")

    config = build_generation_config_from_args(
        _args(
            backend=PYTORCH_GENERATION_BACKEND,
            model="SimpleTransformerLM",
            accelerator="unknown",
            vocab_size=128,
            embedding_dim=32,
            transformer_layers=1,
            attention_heads=4,
            context_length=8,
        )
    )

    assert config.compute_precision == "FP32"
    assert config.quantization == "none"
    assert config.weight_source == "random_weights"
    assert config.accelerator == "CPU"
    assert config.base_url == ""
    assert config.model_cfg.vocab_size == 128
