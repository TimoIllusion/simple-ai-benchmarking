import argparse

from simple_ai_benchmarking.llm_generation import (
    build_generation_config_from_args,
    parse_arguments,
)
from simple_ai_benchmarking.workloads.llm_workload import (
    OLLAMA_BACKEND,
    PYTORCH_GENERATION_BACKEND,
)


def test_saib_llm_runs_default_local_workloads_with_no_args(monkeypatch):
    # Bare `saib-llm` should build the two default local LM workloads (simple
    # transformer and Hugging Face causal LM), mirroring how `saib-pt` runs several
    # models. Low-bit (FP8/FP4) benchmarking is out of scope for the local backends.
    import pytest

    pytest.importorskip("torch")
    from simple_ai_benchmarking.config_pt_tf import get_device_name_pytorch
    from simple_ai_benchmarking.llm_generation import build_generation_configs
    from simple_ai_benchmarking.workloads.llm_workload import HF_CAUSAL_BACKEND

    monkeypatch.setattr("sys.argv", ["saib-llm"])

    args = parse_arguments()
    assert args.backend is None
    configs = build_generation_configs(args)

    assert [c.backend for c in configs] == [
        PYTORCH_GENERATION_BACKEND,
        HF_CAUSAL_BACKEND,
    ]
    # Reference transformer keeps its defaults (self-contained, no server).
    assert configs[0].model == "SimpleTransformerLM"
    assert configs[0].device_name == get_device_name_pytorch()
    assert configs[0].base_url == ""
    # The HF backend defaults to the small Qwen2.5 (portable) at bf16.
    assert configs[1].model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert configs[1].compute_precision == "BF16"
    # Both default workloads put real (batched) load on the device.
    for c in configs:
        assert c.concurrency == 8
        assert c.requests == 32
        assert c.warmup_requests == 2


def test_saib_llm_only_supports_the_remaining_backends():
    # The KV-cache decoder and the torchao FP8/FP4 low-bit backends were removed;
    # low-bit benchmarking now goes through a serving engine (vLLM) over the
    # openai-compatible backend.
    from simple_ai_benchmarking.llm_generation import (
        DEFAULT_LLM_BACKENDS,
        SUPPORTED_LLM_BACKENDS,
    )
    from simple_ai_benchmarking.workloads.llm_workload import (
        HF_CAUSAL_BACKEND,
        OLLAMA_BACKEND,
        OPENAI_COMPATIBLE_BACKEND,
        PYTORCH_GENERATION_BACKEND,
    )

    assert set(SUPPORTED_LLM_BACKENDS) == {
        OPENAI_COMPATIBLE_BACKEND,
        OLLAMA_BACKEND,
        PYTORCH_GENERATION_BACKEND,
        HF_CAUSAL_BACKEND,
    }
    assert DEFAULT_LLM_BACKENDS == (PYTORCH_GENERATION_BACKEND, HF_CAUSAL_BACKEND)
    for removed in ("pytorch-kv-decoder", "huggingface-causal-fp8", "huggingface-causal-fp4"):
        assert removed not in SUPPORTED_LLM_BACKENDS


def test_saib_llm_w_flag_selects_subset_of_default_workloads(monkeypatch):
    import pytest

    pytest.importorskip("torch")
    from simple_ai_benchmarking.llm_generation import build_generation_configs
    from simple_ai_benchmarking.workloads.llm_workload import HF_CAUSAL_BACKEND

    # `-w 1` should run only the second default workload (Hugging Face causal LM).
    monkeypatch.setattr("sys.argv", ["saib-llm", "-w", "1", "--device", "cpu"])
    configs = build_generation_configs(parse_arguments())

    assert [c.backend for c in configs] == [HF_CAUSAL_BACKEND]


def test_saib_llm_w_flag_single_workload(monkeypatch):
    import pytest

    pytest.importorskip("torch")
    from simple_ai_benchmarking.llm_generation import build_generation_configs

    monkeypatch.setattr("sys.argv", ["saib-llm", "-w", "0", "--device", "cpu"])
    configs = build_generation_configs(parse_arguments())

    assert [c.backend for c in configs] == [PYTORCH_GENERATION_BACKEND]


def test_saib_llm_w_flag_rejects_out_of_range(monkeypatch):
    import pytest

    from simple_ai_benchmarking.llm_generation import build_generation_configs

    monkeypatch.setattr("sys.argv", ["saib-llm", "-w", "99"])
    with pytest.raises(SystemExit):
        build_generation_configs(parse_arguments())


def test_saib_llm_w_flag_conflicts_with_backend(monkeypatch):
    import pytest

    monkeypatch.setattr(
        "sys.argv",
        ["saib-llm", "--backend", PYTORCH_GENERATION_BACKEND, "-w", "0"],
    )
    from simple_ai_benchmarking.llm_generation import build_generation_configs

    with pytest.raises(SystemExit):
        build_generation_configs(parse_arguments())


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


def test_llm_publish_each_requires_token(monkeypatch):
    import pytest
    import simple_ai_benchmarking.llm_generation as llm_generation

    monkeypatch.delenv("AI_BENCHMARK_DATABASE_TOKEN", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        ["saib-llm", "--backend", OLLAMA_BACKEND, "--publish-each"],
    )
    monkeypatch.setattr(llm_generation, "build_generation_configs", lambda args: [])

    with pytest.raises(SystemExit, match="requires"):
        llm_generation.run_llm_generation_cli()


def test_llm_publish_each_wires_incremental_publisher(monkeypatch):
    import simple_ai_benchmarking.database as database
    import simple_ai_benchmarking.llm_generation as llm_generation

    publisher = object()
    captured = {}
    monkeypatch.setattr(
        "sys.argv",
        ["saib-llm", "--backend", OLLAMA_BACKEND, "--publish-each", "-t", "tok"],
    )
    monkeypatch.setattr(llm_generation, "build_generation_configs", lambda args: [])
    monkeypatch.setattr(database, "build_incremental_publisher", lambda **kwargs: publisher)
    monkeypatch.setattr(
        llm_generation, "process_workloads", lambda *args, **kwargs: captured.update(kwargs)
    )

    llm_generation.run_llm_generation_cli()

    assert captured["on_workload_logged"] is publisher
