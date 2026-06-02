import pytest

from simple_ai_benchmarking.config_structures import (
    AIFramework,
    AIStage,
    GenerationModelConfig,
    LLMGenerationConfig,
)
from simple_ai_benchmarking.workloads.factory import WorkloadFactory


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
    assert row["bench_info_benchmark_spec_version"] == "2.1"
    assert row["bench_info_benchmark_payload_hash"]
