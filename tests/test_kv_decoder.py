import sys

import pytest

torch = pytest.importorskip("torch")

from simple_ai_benchmarking.config_structures import (
    GenerationModelConfig,
    LLMGenerationConfig,
)
from simple_ai_benchmarking.models.pt.kv_decoder_lm import KVCacheDecoderLM
from simple_ai_benchmarking.workloads.llm_workload import (
    HF_CAUSAL_BACKEND,
    HF_CAUSAL_FP4_BACKEND,
    HF_CAUSAL_FP8_BACKEND,
    PYTORCH_KV_DECODER_BACKEND,
    HuggingFaceCausalFP4Generation,
    HuggingFaceCausalFP8Generation,
    HuggingFaceCausalGeneration,
    PyTorchKVDecoderGeneration,
)


def _skip_if_torchao_available():
    # FP8/int8/int4 only raise NotImplementedError when torchao is absent; when it
    # is installed those paths run real kernels (GPU), so skip the negative tests.
    try:
        import torchao  # noqa: F401

        pytest.skip("torchao installed; low-precision paths run on GPU")
    except ImportError:
        pass


def _tiny_model() -> KVCacheDecoderLM:
    torch.manual_seed(0)
    return KVCacheDecoderLM(
        vocab_size=64,
        context_length=64,
        embedding_dim=32,
        num_heads=4,
        num_layers=2,
        feedforward_dim=64,
    ).eval()


def _reference_generate(model: KVCacheDecoderLM, prompt, generated_tokens):
    # No cache: recompute the full growing sequence each step (slow but obviously
    # correct), used as ground truth for the cached path.
    seq = prompt.clone()
    for _ in range(generated_tokens):
        logits, _ = model.forward(seq)
        next_token = logits[:, -1:, :].argmax(dim=-1)
        seq = torch.cat([seq, next_token], dim=1)
    return seq


def test_kv_cache_matches_full_recompute():
    model = _tiny_model()
    prompt = torch.randint(0, 64, (1, 8))
    cached = model.generate(prompt, 6)
    assert torch.equal(cached, _reference_generate(model, prompt, 6))


def test_batched_kv_cache_matches_full_recompute():
    model = _tiny_model()
    prompt = torch.randint(0, 64, (3, 8))
    cached = model.generate(prompt, 4)
    assert torch.equal(cached, _reference_generate(model, prompt, 4))


def _tiny_kv_config(**overrides) -> LLMGenerationConfig:
    defaults = dict(
        backend=PYTORCH_KV_DECODER_BACKEND,
        device_name="cpu",
        requests=2,
        warmup_requests=0,
        concurrency=1,
        prompt_tokens=4,
        generated_tokens=3,
        context_length=16,
        compute_precision="FP32",
        model_cfg=GenerationModelConfig(
            vocab_size=64,
            context_length=16,
            embedding_dim=32,
            attention_heads=4,
            transformer_layers=2,
            feedforward_dim=64,
        ),
    )
    defaults.update(overrides)
    return LLMGenerationConfig(**defaults)


def test_kv_decoder_workload_runs_and_measures_ttft():
    workload = PyTorchKVDecoderGeneration(_tiny_kv_config())
    workload.setup()
    results, duration = workload._run_measured()

    assert len(results) == 2
    assert duration >= 0
    assert all(r.time_to_first_token_s >= 0 for r in results)
    assert all(r.generated_tokens == 3 for r in results)


def test_kv_decoder_workload_respects_bf16_precision():
    workload = PyTorchKVDecoderGeneration(_tiny_kv_config(compute_precision="BF16"))
    workload.setup()
    assert next(workload._model.parameters()).dtype == torch.bfloat16


def test_kv_decoder_integer_quantization_requires_torchao():
    _skip_if_torchao_available()
    workload = PyTorchKVDecoderGeneration(_tiny_kv_config(quantization="int8"))
    with pytest.raises(NotImplementedError):
        workload.setup()


def test_kv_decoder_fp8_requires_torchao():
    _skip_if_torchao_available()
    workload = PyTorchKVDecoderGeneration(_tiny_kv_config(compute_precision="FP8"))
    with pytest.raises(NotImplementedError):
        workload.setup()


def test_kv_decoder_rejects_unknown_precision():
    workload = PyTorchKVDecoderGeneration(_tiny_kv_config(compute_precision="FP9"))
    with pytest.raises(ValueError):
        workload.setup()


def test_kv_decoder_default_is_about_1b_params():
    # Build on the meta device so no real memory is allocated for the param count.
    with torch.device("meta"):
        model = KVCacheDecoderLM()
    num_params = sum(p.numel() for p in model.parameters())
    assert 0.8e9 < num_params < 1.2e9


def test_kv_decoder_cli_defaults_to_1b_bf16(monkeypatch):
    from simple_ai_benchmarking.llm_generation import (
        build_generation_config_from_args,
        parse_arguments,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["saib-llm", "--backend", PYTORCH_KV_DECODER_BACKEND, "--device", "cpu"],
    )
    config = build_generation_config_from_args(parse_arguments())

    # Fixed ~1B geometry (no presets) and bf16 by default.
    assert config.model_cfg.embedding_dim == 2048
    assert config.model_cfg.transformer_layers == 16
    assert config.model_cfg.attention_heads == 16
    assert config.model_cfg.feedforward_dim == 5632
    assert config.compute_precision == "BF16"
    # Prefill default was raised so time-to-first-token is measurable.
    assert config.prompt_tokens == 2048


def test_hf_workload_runs_on_random_init_model(monkeypatch):
    transformers = pytest.importorskip("transformers")
    from transformers import LlamaConfig

    tiny = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    # Approach B without network: AutoConfig.from_pretrained returns a real config
    # object, then the model is built (random weights) from it.
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a, **k: tiny
    )

    config = LLMGenerationConfig(
        backend=HF_CAUSAL_BACKEND,
        device_name="cpu",
        model="dummy/model",
        requests=2,
        warmup_requests=0,
        concurrency=1,
        prompt_tokens=4,
        generated_tokens=3,
        compute_precision="FP32",
    )
    workload = HuggingFaceCausalGeneration(config)
    workload.setup()
    results, duration = workload._run_measured()

    assert len(results) == 2
    assert duration >= 0
    assert all(r.generated_tokens == 3 for r in results)
    assert all(r.time_to_first_token_s >= 0 for r in results)


def test_hf_workload_integer_quantization_requires_torchao(monkeypatch):
    transformers = pytest.importorskip("transformers")
    _skip_if_torchao_available()
    from transformers import LlamaConfig

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *a, **k: LlamaConfig(
            vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2
        ),
    )
    config = LLMGenerationConfig(
        backend=HF_CAUSAL_BACKEND,
        device_name="cpu",
        model="dummy/model",
        compute_precision="FP32",
        quantization="int4",
    )
    with pytest.raises(NotImplementedError):
        HuggingFaceCausalGeneration(config).setup()


def test_hf_workload_clear_error_when_transformers_unusable(monkeypatch):
    # Simulate transformers being importable but lacking the names (the same
    # ImportError class raised when transformers 5.x is incompatible with torch):
    # the backend should fail with an actionable NotImplementedError, not a cryptic
    # "Could not import module 'AutoModelForCausalLM'".
    import types

    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    config = LLMGenerationConfig(
        backend=HF_CAUSAL_BACKEND,
        device_name="cpu",
        model="dummy/model",
        compute_precision="FP32",
    )
    with pytest.raises(NotImplementedError, match="transformers"):
        HuggingFaceCausalGeneration(config).setup()


def _mock_tiny_hf_config(monkeypatch):
    transformers = pytest.importorskip("transformers")
    from transformers import LlamaConfig

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *a, **k: LlamaConfig(
            vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2
        ),
    )


def test_hf_fp8_low_bit_backend_identity_and_requires_torchao(monkeypatch):
    _mock_tiny_hf_config(monkeypatch)
    config = LLMGenerationConfig(
        backend=HF_CAUSAL_FP8_BACKEND,
        device_name="cpu",
        model="dummy/model",
        compute_precision="FP8",
    )
    workload = HuggingFaceCausalFP8Generation(config)
    # Distinct backend identity so FP8 results are comparable on their own.
    assert workload._get_backend() == HF_CAUSAL_FP8_BACKEND
    _skip_if_torchao_available()
    with pytest.raises(NotImplementedError):
        workload.setup()


def test_hf_fp4_low_bit_backend_identity_and_requires_torchao(monkeypatch):
    _mock_tiny_hf_config(monkeypatch)
    config = LLMGenerationConfig(
        backend=HF_CAUSAL_FP4_BACKEND,
        device_name="cpu",
        model="dummy/model",
        compute_precision="FP4",
    )
    workload = HuggingFaceCausalFP4Generation(config)
    assert workload._get_backend() == HF_CAUSAL_FP4_BACKEND
    _skip_if_torchao_available()
    with pytest.raises(NotImplementedError):
        workload.setup()


def test_saib_llm_low_bit_workloads_pin_precision_and_share_model(monkeypatch):
    from simple_ai_benchmarking.llm_generation import (
        build_generation_config_from_args,
        parse_arguments,
    )

    for backend, expected_precision in (
        (HF_CAUSAL_FP8_BACKEND, "FP8"),
        (HF_CAUSAL_FP4_BACKEND, "FP4"),
    ):
        monkeypatch.setattr(
            sys, "argv", ["saib-llm", "--backend", backend, "--device", "cpu"]
        )
        config = build_generation_config_from_args(parse_arguments())
        # Low-bit variants reuse the HF default repo and pin their precision.
        assert config.backend == backend
        assert config.model  # substituted to the default HF repo id
        assert config.compute_precision == expected_precision
