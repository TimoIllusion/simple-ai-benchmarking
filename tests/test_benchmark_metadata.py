from simple_ai_benchmarking.benchmark_metadata import (
    build_cv_profile,
    build_llm_profile,
    canonical_hash,
)
from simple_ai_benchmarking.llm_results import (
    LLMBenchInfo,
    LLMBenchmarkLogger,
    LLMBenchmarkResult,
    LLMPerformanceResult,
)
from simple_ai_benchmarking.results import (
    BaseBenchmarkLogger,
    BenchInfo,
    BenchmarkLogger,
    BenchmarkResult,
    HWInfo,
    PerformanceResult,
    SWInfo,
    collect_hw_info,
    collect_sw_info,
)


def test_canonical_hash_is_stable_across_dict_ordering():
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})


def test_cv_and_llm_loggers_share_base_export_machinery():
    assert issubclass(BenchmarkLogger, BaseBenchmarkLogger)
    assert issubclass(LLMBenchmarkLogger, BaseBenchmarkLogger)


def test_collect_sw_info_populates_host_runtime_fields():
    sw_info = collect_sw_info("torch", "2.4.0", "cuda")

    assert sw_info.ai_framework_name == "torch"
    assert sw_info.ai_framework_version == "2.4.0"
    assert sw_info.ai_framework_extra_info == "cuda"
    assert sw_info.python_version
    assert sw_info.os_version


def test_collect_hw_info_uses_passed_accelerator_and_host_fields():
    hw_info = collect_hw_info("NVIDIA RTX 4090")

    assert hw_info.accelerator == "NVIDIA RTX 4090"
    assert hw_info.cpu
    assert hw_info.num_cores >= 1
    assert hw_info.ram_gb > 0


def test_cv_profile_hash_changes_for_workload_parameters_only():
    base = build_cv_profile(
        workload_type="InferenceWorkload",
        model="ResNet50",
        compute_precision="DEFAULT_PRECISION",
        batch_size=32,
        sample_shape=[224, 224, 3],
        num_classes=1000,
    )
    changed_batch = dict(base, batch_size=64)
    with_runtime_noise = dict(
        base,
        benchmark_version="0.4.1",
        benchmark_date="2026-05-11T12:00:00",
        hardware="NVIDIA RTX 4090",
    )

    assert canonical_hash(base) != canonical_hash(changed_batch)
    assert canonical_hash(base) == canonical_hash(
        {key: with_runtime_noise[key] for key in base}
    )


def test_llm_profile_hash_changes_for_comparability_parameters():
    base = build_llm_profile(
        backend="llama.cpp",
        model="Meta-Llama-3-8B",
        benchmark_type="inference",
        compute_precision="FP16",
        quantization="Q4_K_M",
        context_length=4096,
        prompt_tokens=128,
        generated_tokens=256,
        concurrency=1,
    )

    for key, value in {
        "prompt_token_target": 256,
        "generated_token_target": 512,
        "context_length": 8192,
        "concurrency": 2,
        "backend_protocol_class": "openai-compatible",
    }.items():
        assert canonical_hash(base) != canonical_hash(dict(base, **{key: value}))


def test_cv_csv_export_contains_benchmark_metadata():
    bench_info = BenchInfo(
        workload_type="InferenceWorkload",
        model="ResNet50",
        compute_precision="DEFAULT_PRECISION",
        batch_size=32,
        date="2026-05-11T12:00:00",
        sample_shape=[224, 224, 3],
        num_classes=1000,
        num_parameters=25500000,
    )
    performance = PerformanceResult(iterations=10)
    performance.update_duration_and_calc_throughput(1.0)
    logger = BenchmarkLogger()
    logger.add_result(
        BenchmarkResult(
            sw_info=SWInfo("PyTorch", "2.4.0", "cuda", "3.11.0", "Linux"),
            hw_info=HWInfo("AMD Ryzen", 16, 64.0, "NVIDIA RTX 4090"),
            bench_info=bench_info,
            performance=performance,
        )
    )

    row = logger.to_dataframe().iloc[0].to_dict()

    assert row["bench_info_benchmark_family"] == "cv"
    assert row["bench_info_benchmark_profile_hash"]
    assert row["bench_info_benchmark_payload_hash"]


def test_llm_csv_export_contains_benchmark_metadata():
    logger = LLMBenchmarkLogger()
    logger.add_result(
        LLMBenchmarkResult(
            sw_info=SWInfo("llama.cpp", "0.0.1", "cuda", "3.11.0", "Linux"),
            hw_info=HWInfo("AMD Ryzen", 16, 64.0, "NVIDIA RTX 4090"),
            bench_info=LLMBenchInfo(
                benchmark_type="inference",
                backend="llama.cpp",
                model="Meta-Llama-3-8B",
                model_params=8000000000,
                compute_precision="FP16",
                quantization="Q4_K_M",
                context_length=4096,
                prompt_tokens=128,
                generated_tokens=256,
                concurrency=1,
                date="2026-05-11T12:00:00",
                weight_source="random_weights",
            ),
            performance=LLMPerformanceResult(
                requests=10,
                duration_s=60.0,
                prompt_tokens_per_second=21.3,
                generated_tokens_per_second=42.5,
                total_tokens_per_second=63.8,
                time_to_first_token_s=0.24,
            ),
        )
    )

    row = logger.to_dataframe().iloc[0].to_dict()

    assert row["bench_info_benchmark_family"] == "llm"
    assert row["bench_info_weight_source"] == "random_weights"
    assert row["bench_info_benchmark_profile_hash"]
    assert row["bench_info_benchmark_payload_hash"]
