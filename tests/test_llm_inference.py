from simple_ai_benchmarking.llm_inference import (
    LLMBackendClient,
    LLMInferenceBenchmark,
    LLMInferenceConfig,
    OLLAMA_BACKEND,
    OPENAI_COMPATIBLE_BACKEND,
    PYTORCH_SIMPLE_TRANSFORMER_BACKEND,
    build_config_from_args,
)


class FakeResponse:
    def __init__(self, lines):
        self.lines = lines

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(self.lines)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeSession:
    def __init__(self, lines):
        self.lines = lines
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse(self.lines)


def test_openai_compatible_backend_posts_chat_completion_request():
    session = FakeSession(
        [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":17,"completion_tokens":23}}',
            "data: [DONE]",
        ]
    )
    config = LLMInferenceConfig(
        backend=OPENAI_COMPATIBLE_BACKEND,
        base_url="https://example.test",
        model="test-model",
        api_key="token",
        generated_tokens=23,
    )
    client = LLMBackendClient(config, session=session)

    result = client.generate("hello")

    assert session.calls[0][0] == "https://example.test/v1/chat/completions"
    assert session.calls[0][1]["headers"]["Authorization"] == "Bearer token"
    assert session.calls[0][1]["json"]["model"] == "test-model"
    assert session.calls[0][1]["json"]["stream"] is True
    assert result.prompt_tokens == 17
    assert result.generated_tokens == 23
    assert 0 < result.time_to_first_token_s <= result.duration_s


def test_ollama_backend_posts_generate_request():
    session = FakeSession(
        [
            '{"response":"Hel"}',
            '{"response":"lo"}',
            '{"done":true,"eval_count":19,"prompt_eval_count":11}',
        ]
    )
    config = LLMInferenceConfig(
        backend=OLLAMA_BACKEND,
        base_url="http://localhost:11434",
        model="llama3",
        generated_tokens=19,
    )
    client = LLMBackendClient(config, session=session)

    result = client.generate("hello")

    assert session.calls[0][0] == "http://localhost:11434/api/generate"
    assert session.calls[0][1]["json"]["options"]["num_predict"] == 19
    assert session.calls[0][1]["json"]["stream"] is True
    assert result.prompt_tokens == 11
    assert result.generated_tokens == 19
    assert 0 < result.time_to_first_token_s <= result.duration_s


def test_llm_inference_benchmark_builds_exportable_result():
    session = FakeSession(
        [
            'data: {"choices":[{"delta":{"content":"a"}}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20}}',
            "data: [DONE]",
        ]
    )
    config = LLMInferenceConfig(
        backend=OPENAI_COMPATIBLE_BACKEND,
        base_url="https://example.test",
        model="test-model",
        requests=2,
        warmup_requests=1,
        prompt_tokens=10,
        generated_tokens=20,
        concurrency=1,
    )
    benchmark = LLMInferenceBenchmark(config, LLMBackendClient(config, session))

    result = benchmark.run()

    assert len(session.calls) == 3
    assert result.bench_info.backend == OPENAI_COMPATIBLE_BACKEND
    assert result.bench_info.model == "test-model"
    assert result.performance.requests == 2
    assert result.performance.generated_tokens_per_second > 0
    assert result.bench_info.benchmark_profile_hash


def test_pytorch_simple_transformer_backend_generates_tokens():
    import pytest

    torch = pytest.importorskip("torch")
    from simple_ai_benchmarking.models.pt.simple_transformer_lm import (
        SimpleTransformerLanguageModel,
    )

    model = SimpleTransformerLanguageModel(
        vocab_size=128,
        context_length=16,
        embedding_dim=32,
        num_heads=4,
        num_layers=1,
        feedforward_dim=64,
    )
    input_ids = torch.arange(8, dtype=torch.long).unsqueeze(0)

    output_ids = model.generate(input_ids, generated_tokens=4)

    assert output_ids.shape == (1, 12)


def test_pytorch_simple_transformer_backend_builds_result():
    import pytest

    pytest.importorskip("torch")
    config = LLMInferenceConfig(
        backend=PYTORCH_SIMPLE_TRANSFORMER_BACKEND,
        base_url="",
        model="SimpleTransformerLM",
        requests=1,
        warmup_requests=0,
        prompt_tokens=4,
        generated_tokens=2,
        context_length=8,
        device="cpu",
        vocab_size=128,
        embedding_dim=32,
        transformer_layers=1,
        attention_heads=4,
    )
    benchmark = LLMInferenceBenchmark(config)

    result = benchmark.run()

    assert result.bench_info.backend == PYTORCH_SIMPLE_TRANSFORMER_BACKEND
    assert result.bench_info.model == "SimpleTransformerLM"
    assert result.bench_info.model_params > 0
    assert result.performance.generated_tokens_per_second > 0
    assert (
        0
        < result.performance.time_to_first_token_s
        <= result.performance.duration_s
    )


def test_build_config_uses_backend_default_base_urls(monkeypatch):
    class Args:
        backend = OLLAMA_BACKEND
        base_url = None
        model = "llama3"
        requests = 1
        warmup_requests = 0
        concurrency = 1
        prompt_tokens = 1
        generated_tokens = 1
        context_length = 2048
        timeout_s = 1.0
        api_key = None
        api_key_env = "OPENAI_API_KEY"
        compute_precision = ""
        quantization = ""
        model_params = 0
        ai_framework_version = ""
        ai_framework_extra_info = ""
        accelerator = "cpu"
        weight_source = ""
        device = "cpu"
        vocab_size = 32000
        embedding_dim = 256
        transformer_layers = 4
        attention_heads = 4

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test")

    config = build_config_from_args(Args())

    assert config.base_url == "http://ollama.test"


def test_pytorch_simple_transformer_metadata_separates_quant_and_weights():
    import pytest

    pytest.importorskip("torch")

    class Args:
        backend = PYTORCH_SIMPLE_TRANSFORMER_BACKEND
        base_url = None
        model = "SimpleTransformerLM"
        requests = 1
        warmup_requests = 0
        concurrency = 1
        prompt_tokens = 1
        generated_tokens = 1
        context_length = 8
        timeout_s = 1.0
        api_key = None
        api_key_env = "OPENAI_API_KEY"
        compute_precision = ""
        quantization = ""
        model_params = 0
        ai_framework_version = ""
        ai_framework_extra_info = ""
        accelerator = "cpu"
        weight_source = ""
        device = "cpu"
        vocab_size = 128
        embedding_dim = 32
        transformer_layers = 1
        attention_heads = 4

    config = build_config_from_args(Args())

    assert config.compute_precision == "FP32"
    assert config.quantization == "none"
    assert config.weight_source == "random_weights"
