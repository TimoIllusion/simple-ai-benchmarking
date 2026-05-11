# Project Name: simple-ai-benchmarking
# File Name: llm_results.py

from dataclasses import asdict, dataclass, field
from typing import List

import pandas as pd
from tabulate import tabulate

from simple_ai_benchmarking.benchmark_metadata import (
    LLM_BENCHMARK_FAMILY,
    LLM_RUNNER_ID,
    LLM_SPEC_NAME,
    SPEC_VERSION,
    build_llm_profile,
    build_llm_profile_id,
    build_payload_hash,
    build_runner_hash,
    canonical_hash,
)
from simple_ai_benchmarking.results import HWInfo, SWInfo


@dataclass
class LLMBenchInfo:
    benchmark_type: str
    backend: str
    model: str
    model_params: int
    compute_precision: str
    quantization: str
    context_length: int
    prompt_tokens: int
    generated_tokens: int
    concurrency: int
    date: str
    benchmark_family: str = field(init=False)
    benchmark_spec_name: str = field(init=False)
    benchmark_spec_version: str = field(init=False)
    benchmark_profile_id: str = field(init=False)
    benchmark_profile_hash: str = field(init=False)
    benchmark_config_hash: str = field(init=False)
    benchmark_runner_id: str = field(init=False)
    benchmark_runner_hash: str = field(init=False)
    benchmark_payload_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        profile = build_llm_profile(
            backend=self.backend,
            model=self.model,
            benchmark_type=self.benchmark_type,
            compute_precision=self.compute_precision,
            quantization=self.quantization,
            context_length=self.context_length,
            prompt_tokens=self.prompt_tokens,
            generated_tokens=self.generated_tokens,
            concurrency=self.concurrency,
        )
        self.benchmark_family = LLM_BENCHMARK_FAMILY
        self.benchmark_spec_name = LLM_SPEC_NAME
        self.benchmark_spec_version = SPEC_VERSION
        self.benchmark_profile_id = build_llm_profile_id(profile)
        self.benchmark_profile_hash = canonical_hash(profile)
        self.benchmark_config_hash = canonical_hash(profile)
        self.benchmark_runner_id = LLM_RUNNER_ID
        self.benchmark_runner_hash = build_runner_hash(LLM_RUNNER_ID)


@dataclass
class LLMPerformanceResult:
    requests: int
    duration_s: float
    prompt_tokens_per_second: float
    generated_tokens_per_second: float
    total_tokens_per_second: float
    time_to_first_token_s: float
    finished_successfully: bool = True
    error_message: str = ""


@dataclass
class LLMBenchmarkResult:
    sw_info: SWInfo
    hw_info: HWInfo
    bench_info: LLMBenchInfo
    performance: LLMPerformanceResult


class LLMBenchmarkLogger:
    def __init__(self) -> None:
        self.results: List[LLMBenchmarkResult] = []

    def add_result(self, result: LLMBenchmarkResult) -> None:
        self.results.append(result)

    def to_dataframe(self) -> pd.DataFrame:
        flat_dicts = []
        for result in self.results:
            flat_dict = {}
            for key, value in asdict(result).items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat_dict[f"{key}_{sub_key}"] = sub_value
                else:
                    flat_dict[key] = value
            flat_dict["bench_info_benchmark_payload_hash"] = build_payload_hash(
                flat_dict
            )
            flat_dicts.append(flat_dict)

        return pd.DataFrame(flat_dicts)

    def pretty_print_summary(self) -> None:
        print("\n===== LLM BENCHMARK SUMMARY =====\n")

        header = [
            "#RUN",
            "Backend",
            "Model",
            "Accelerator",
            "Quant",
            "Ctx",
            "Conc",
            "Gen tok/s",
            "TTFT s",
        ]
        table_data = []

        for i, result in enumerate(self.results):
            table_data.append(
                [
                    str(i),
                    result.bench_info.backend,
                    result.bench_info.model,
                    result.hw_info.accelerator,
                    result.bench_info.quantization,
                    result.bench_info.context_length,
                    result.bench_info.concurrency,
                    round(result.performance.generated_tokens_per_second, 2),
                    round(result.performance.time_to_first_token_s, 3),
                ]
            )

        print(tabulate(table_data, headers=header, tablefmt="pretty"))

    def export_to_csv(self, file_name: str) -> None:
        self.to_dataframe().to_csv(file_name, index=False)

    def export_to_excel(self, file_name: str) -> None:
        self.to_dataframe().to_excel(file_name, index=False)
