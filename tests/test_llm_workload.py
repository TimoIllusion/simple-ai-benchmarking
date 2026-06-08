import pytest

from simple_ai_benchmarking.config_structures import (
    AIFramework,
    AIStage,
    GenerationModelConfig,
    LLMGenerationConfig,
)
from simple_ai_benchmarking.workloads.factory import WorkloadFactory
from simple_ai_benchmarking.workloads.llm_workload import (
    OLLAMA_BACKEND,
    OPENAI_COMPATIBLE_BACKEND,
    VLLM_SERVING_ENGINE,
    OllamaGeneration,
    OpenAICompatibleGeneration,
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


class FakeVersionResponse:
    def __init__(self, payload, ok=True):
        self._payload = payload
        self.ok = ok

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, lines, version="0.6.3"):
        self.lines = lines
        self.version = version
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse(self.lines)

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeVersionResponse({"version": self.version})


def _tiny_config(**overrides) -> LLMGenerationConfig:
    cfg = LLMGenerationConfig(
        device_name="cpu",
        model="SimpleTransformerLM",
        requests=4,
        warmup_requests=0,
        concurrency=2,
        prompt_tokens=4,
        generated_tokens=2,
        model_cfg=GenerationModelConfig(
            vocab_size=128,
            context_length=8,
            embedding_dim=32,
            attention_heads=4,
            transformer_layers=1,
        ),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_factory_builds_pytorch_generation_workload():
    pytest.importorskip("torch")
    from simple_ai_benchmarking.workloads.llm_workload import PyTorchLocalGeneration

    workload = WorkloadFactory.create_workload(_tiny_config(), AIFramework.PYTORCH)

    assert isinstance(workload, PyTorchLocalGeneration)
    assert workload._get_ai_stage() is AIStage.GENERATION


def test_generation_workload_runs_lifecycle_and_builds_result():
    pytest.importorskip("torch")
    from simple_ai_benchmarking.workloads.llm_workload import PyTorchLocalGeneration

    workload = PyTorchLocalGeneration(_tiny_config(warmup_requests=1))
    workload.setup()
    workload.warmup()
    workload.prepare_execution()
    workload.execute()
    result = workload.build_result_log()

    assert result.bench_info.backend == "pytorch-simple-transformer"
    assert result.bench_info.model == "SimpleTransformerLM"
    assert result.bench_info.concurrency == 2
    assert result.bench_info.model_params > 0
    assert result.performance.requests == 4
    assert result.performance.generated_tokens_per_second > 0
    assert 0 < result.performance.time_to_first_token_s <= result.performance.duration_s


def test_generation_concurrency_is_batch_size():
    pytest.importorskip("torch")
    from simple_ai_benchmarking.workloads.llm_workload import (
        GenerationRequestResult,
        PyTorchLocalGeneration,
    )

    workload = PyTorchLocalGeneration(_tiny_config(requests=5, concurrency=2))
    workload.setup()

    batch_sizes = []
    original_generate_batch = workload._generate_batch

    def spy(batch_size):
        batch_sizes.append(batch_size)
        return original_generate_batch(batch_size)

    workload._generate_batch = spy
    request_results, duration_s = workload._run_measured()

    # 5 requests with batch size (concurrency) 2 -> batches of 2, 2, 1.
    assert batch_sizes == [2, 2, 1]
    assert len(request_results) == 5
    assert all(isinstance(r, GenerationRequestResult) for r in request_results)
    assert duration_s > 0


def test_factory_dispatches_http_backends_without_torch():
    openai_cfg = LLMGenerationConfig(
        backend=OPENAI_COMPATIBLE_BACKEND, base_url="https://example.test"
    )
    ollama_cfg = LLMGenerationConfig(
        backend=OLLAMA_BACKEND, base_url="http://localhost:11434"
    )

    openai_workload = WorkloadFactory.create_workload(openai_cfg, AIFramework.PYTORCH)
    ollama_workload = WorkloadFactory.create_workload(ollama_cfg, AIFramework.PYTORCH)

    assert isinstance(openai_workload, OpenAICompatibleGeneration)
    assert isinstance(ollama_workload, OllamaGeneration)


def test_openai_compatible_generation_workload_streams_and_counts_tokens():
    workload = OpenAICompatibleGeneration(
        LLMGenerationConfig(
            backend=OPENAI_COMPATIBLE_BACKEND,
            base_url="https://example.test",
            model="test-model",
            api_key="token",
            requests=1,
            warmup_requests=0,
            concurrency=1,
            generated_tokens=23,
        )
    )
    workload._session = FakeSession(
        [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":17,"completion_tokens":23}}',
            "data: [DONE]",
        ]
    )
    workload.setup()
    workload.warmup()
    workload.execute()
    result = workload.build_result_log()

    assert workload._session.calls[0][0] == "https://example.test/v1/chat/completions"
    assert workload._session.calls[0][1]["headers"]["Authorization"] == "Bearer token"
    assert result.bench_info.backend == OPENAI_COMPATIBLE_BACKEND
    assert result.performance.generated_tokens_per_second > 0


def test_openai_compatible_uses_server_version_as_framework_version():
    # Regression: vLLM/openai-compatible runs left ai_framework_version blank, which
    # the database rejects ("This field may not be blank."). The version must be
    # resolved from the engine's GET /version so results publish.
    workload = OpenAICompatibleGeneration(
        LLMGenerationConfig(
            backend=OPENAI_COMPATIBLE_BACKEND,
            base_url="https://example.test",
            model="test-model",
            served_by="vllm",
            requests=1,
            warmup_requests=0,
            concurrency=1,
            generated_tokens=2,
        )
    )
    workload._session = FakeSession(
        [
            'data: {"choices":[{"delta":{"content":"hi"}}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}',
            "data: [DONE]",
        ],
        version="0.6.3.post1",
    )
    workload.setup()
    workload.warmup()
    workload.execute()
    result = workload.build_result_log()

    assert any(c[0] == "https://example.test/version" for c in workload._session.calls)
    assert result.sw_info.ai_framework_version == "0.6.3.post1"


def test_openai_compatible_vllm_resolves_model_params(monkeypatch):
    import simple_ai_benchmarking.workloads.llm_workload as llm_workload

    seen = {}

    def fake_count(model_id):
        seen["model_id"] = model_id
        return 123456789

    monkeypatch.setattr(
        llm_workload, "_count_huggingface_causal_lm_parameters", fake_count
    )
    workload = OpenAICompatibleGeneration(
        LLMGenerationConfig(
            backend=OPENAI_COMPATIBLE_BACKEND,
            base_url="https://example.test",
            model="org/model",
            served_by=VLLM_SERVING_ENGINE,
            requests=1,
            warmup_requests=0,
            concurrency=1,
            generated_tokens=2,
        )
    )
    workload._session = FakeSession(
        [
            'data: {"choices":[{"delta":{"content":"hi"}}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}',
            "data: [DONE]",
        ]
    )
    workload.setup()
    workload.warmup()
    workload.execute()
    result = workload.build_result_log()

    assert seen["model_id"] == "org/model"
    assert result.bench_info.model_params == 123456789


def test_openai_compatible_keeps_explicit_model_params(monkeypatch):
    import simple_ai_benchmarking.workloads.llm_workload as llm_workload

    def fail_if_called(model_id):
        raise AssertionError(f"unexpected resolver call for {model_id}")

    monkeypatch.setattr(
        llm_workload, "_count_huggingface_causal_lm_parameters", fail_if_called
    )
    workload = OpenAICompatibleGeneration(
        LLMGenerationConfig(
            backend=OPENAI_COMPATIBLE_BACKEND,
            base_url="https://example.test",
            model="org/model",
            served_by=VLLM_SERVING_ENGINE,
            model_params=42,
        )
    )
    workload._session = FakeSession([])
    workload.setup()

    assert workload.cfg.model_params == 42


def test_openai_compatible_non_vllm_does_not_resolve_model_params(monkeypatch):
    import simple_ai_benchmarking.workloads.llm_workload as llm_workload

    def fail_if_called(model_id):
        raise AssertionError(f"unexpected resolver call for {model_id}")

    monkeypatch.setattr(
        llm_workload, "_count_huggingface_causal_lm_parameters", fail_if_called
    )
    workload = OpenAICompatibleGeneration(
        LLMGenerationConfig(
            backend=OPENAI_COMPATIBLE_BACKEND,
            base_url="https://example.test",
            model="gpt-test",
            served_by="openai",
        )
    )
    workload._session = FakeSession([])
    workload.setup()

    assert workload.cfg.model_params == 0


def test_openai_compatible_falls_back_to_served_by_when_no_version():
    # If the engine has no /version endpoint, the identity field must still be
    # non-empty so the row is accepted; fall back to the serving-engine label.
    workload = OpenAICompatibleGeneration(
        LLMGenerationConfig(
            backend=OPENAI_COMPATIBLE_BACKEND,
            base_url="https://example.test",
            model="test-model",
            served_by="vllm",
            requests=1,
            warmup_requests=0,
            concurrency=1,
            generated_tokens=2,
        )
    )

    class NoVersionSession(FakeSession):
        def get(self, url, **kwargs):
            raise RuntimeError("no /version endpoint")

    workload._session = NoVersionSession(
        [
            'data: {"choices":[{"delta":{"content":"hi"}}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}',
            "data: [DONE]",
        ]
    )
    workload.setup()
    workload.warmup()
    workload.execute()
    result = workload.build_result_log()

    assert result.sw_info.ai_framework_version == "vllm"


def test_openai_compatible_reports_zero_when_server_streams_no_tokens():
    # Regression: an empty/refused completion (no content deltas, no usage) must
    # report 0 generated tokens, not fabricate the requested count.
    workload = OpenAICompatibleGeneration(
        LLMGenerationConfig(
            backend=OPENAI_COMPATIBLE_BACKEND,
            base_url="https://example.test",
            model="test-model",
            requests=1,
            warmup_requests=0,
            concurrency=1,
            generated_tokens=23,
        )
    )
    workload._session = FakeSession(["data: [DONE]"])
    workload.setup()
    workload.warmup()
    workload.execute()
    result = workload.build_result_log()

    assert result.performance.generated_tokens_per_second == 0
    assert 0 < result.performance.time_to_first_token_s <= result.performance.duration_s


def test_ollama_generation_workload_streams_and_counts_tokens():
    workload = OllamaGeneration(
        LLMGenerationConfig(
            backend=OLLAMA_BACKEND,
            base_url="http://localhost:11434",
            model="llama3",
            requests=1,
            warmup_requests=0,
            concurrency=1,
            generated_tokens=19,
        )
    )
    workload._session = FakeSession(
        [
            '{"response":"Hel"}',
            '{"response":"lo"}',
            '{"done":true,"eval_count":19,"prompt_eval_count":11}',
        ]
    )
    workload.setup()
    workload.warmup()
    workload.execute()
    result = workload.build_result_log()

    assert workload._session.calls[0][0] == "http://localhost:11434/api/generate"
    assert workload._session.calls[0][1]["json"]["options"]["num_predict"] == 19
    assert result.bench_info.backend == OLLAMA_BACKEND
    assert result.performance.generated_tokens_per_second > 0


def test_generation_workload_result_is_exportable():
    pytest.importorskip("torch")
    from simple_ai_benchmarking.llm_results import LLMBenchmarkLogger
    from simple_ai_benchmarking.workloads.llm_workload import PyTorchLocalGeneration

    workload = PyTorchLocalGeneration(_tiny_config())
    workload.setup()
    workload.warmup()
    workload.execute()
    result = workload.build_result_log()

    logger = LLMBenchmarkLogger()
    logger.add_result(result)
    row = logger.to_dataframe().iloc[0].to_dict()

    assert row["bench_info_benchmark_family"] == "llm"
    assert row["bench_info_benchmark_spec_version"] == "2.3"
    assert row["bench_info_benchmark_payload_hash"]
    # The local PyTorch backend records its engine as "pytorch".
    assert row["bench_info_serving_engine"] == "pytorch"
