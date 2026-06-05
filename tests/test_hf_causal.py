import sys
import types

import pytest

torch = pytest.importorskip("torch")

from simple_ai_benchmarking.config_structures import LLMGenerationConfig
from simple_ai_benchmarking.workloads.llm_workload import (
    HF_CAUSAL_BACKEND,
    HuggingFaceCausalGeneration,
)


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


def test_hf_workload_resolves_config_from_raw_repo_id(monkeypatch):
    # Bundling was removed: the backend must fetch the config straight from the
    # given repo id (no local-dir substitution).
    transformers = pytest.importorskip("transformers")
    from transformers import LlamaConfig

    seen = {}

    def fake_from_pretrained(model_id, *a, **k):
        seen["model_id"] = model_id
        return LlamaConfig(
            vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2
        )

    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", fake_from_pretrained
    )
    config = LLMGenerationConfig(
        backend=HF_CAUSAL_BACKEND,
        device_name="cpu",
        model="Qwen/Qwen3-1.7B",
        compute_precision="BF16",
    )
    HuggingFaceCausalGeneration(config).setup()
    assert seen["model_id"] == "Qwen/Qwen3-1.7B"


def test_hf_workload_rejects_unknown_precision(monkeypatch):
    transformers = pytest.importorskip("transformers")
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
        compute_precision="FP8",  # low-bit is out of scope for the local backend
    )
    with pytest.raises(ValueError, match="compute precision"):
        HuggingFaceCausalGeneration(config).setup()


def test_hf_workload_clear_error_when_transformers_unusable(monkeypatch):
    # Simulate transformers being importable but lacking the names (the same
    # ImportError class raised when transformers 5.x is incompatible with torch):
    # the backend should fail with an actionable NotImplementedError, not a cryptic
    # "Could not import module 'AutoModelForCausalLM'".
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    config = LLMGenerationConfig(
        backend=HF_CAUSAL_BACKEND,
        device_name="cpu",
        model="dummy/model",
        compute_precision="FP32",
    )
    with pytest.raises(NotImplementedError, match="transformers"):
        HuggingFaceCausalGeneration(config).setup()
