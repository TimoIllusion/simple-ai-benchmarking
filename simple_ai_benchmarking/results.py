from dataclasses import dataclass, field, asdict
from typing import List
import sys

from loguru import logger

import pandas as pd
from tabulate import tabulate


def initialize_logger(log_file: str) -> None:
    logger.remove()
    logger.add(log_file, rotation="10 MB", backtrace=True)
    logger.add(sys.stdout, colorize=True, backtrace=True, level="INFO")


@dataclass
class SWInfo:
    ai_framework_name: str
    ai_framework_version: str
    ai_framework_extra_info: str
    python_version: str
    os_version: str


@dataclass
class HWInfo:
    cpu: str
    num_cores: int
    ram_gb: float
    accelerator: str


@dataclass
class PerformanceResult:
    iterations: int
    duration_s: float = field(init=False)
    throughput: float = field(init=False)
    finished_successfully: bool = True
    error_message: str = ""

    def update_duration_and_calc_throughput(self, duration_s: float) -> None:

        self.update_duration(duration_s)
        self.calc_throughput_and_update()

    def update_duration(self, duration_s: float) -> float:
        self.duration_s = duration_s

    def calc_throughput_and_update(self) -> None:
        self.throughput = (
            self.iterations / self.duration_s if self.duration_s > 0 else 0.0
        )


@dataclass
class BenchInfo:
    workload_type: str
    model: str
    compute_precision: str
    batch_size_training: int
    batch_size_inference: int
    date: str
    sample_shape: List[int]


@dataclass
class BenchmarkResult:
    sw_info: SWInfo
    hw_info: HWInfo
    bench_info: BenchInfo
    train_performance: PerformanceResult
    infer_performance: PerformanceResult

    def update_train_performance_duration(self, duration_s: float) -> None:
        self.train_performance.update_duration_and_calc_throughput(duration_s)

    def update_infer_performance_duration(self, duration_s: float) -> None:
        self.infer_performance.update_duration_and_calc_throughput(duration_s)


class BenchmarkLogger:

    def __init__(self) -> None:
        self.results: List[BenchmarkResult] = []

    def add_repetitions_for_one_benchmark(
        self, repetition_results: List[BenchmarkResult]
    ) -> None:

        averaged_benchmark_result = self._average_benchmark_results(repetition_results)
        self.add_result(averaged_benchmark_result)

    def _average_benchmark_results(
        self, benchmark_results: List[BenchmarkResult]
    ) -> BenchmarkResult:

        assert benchmark_results, "Got empty list of benchmark results"

        infer_performances = [result.infer_performance for result in benchmark_results]
        train_performances = [result.train_performance for result in benchmark_results]

        infer_avg_perf = self._accumulate_and_average_performance_results(
            infer_performances
        )
        train_avg_perf = self._accumulate_and_average_performance_results(
            train_performances
        )

        combined_avg_benchmark_result = benchmark_results[0]
        combined_avg_benchmark_result.infer_performance = infer_avg_perf
        combined_avg_benchmark_result.train_performance = train_avg_perf

        return combined_avg_benchmark_result

    def _accumulate_and_average_performance_results(
        self, perf_results: List[PerformanceResult]
    ) -> PerformanceResult:

        iterations_sum = 0
        duration_s_sum = 0.0

        for perf in perf_results:
            iterations_sum += perf.iterations
            duration_s_sum += perf.duration_s

        avg_result = PerformanceResult(iterations_sum)
        avg_result.update_duration_and_calc_throughput(duration_s_sum)

        return avg_result

    def add_result(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    def to_dataframe(self) -> pd.DataFrame:

        # Convert each BenchmarkResult to a nested dictionary
        nested_dicts = [asdict(result) for result in self.results]

        # Flatten the nested dictionaries for Pandas
        flat_dicts = []
        for nested_dict in nested_dicts:
            flat_dict = {}
            for key, value in nested_dict.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat_dict[f"{key}_{sub_key}"] = sub_value
                else:
                    flat_dict[key] = value
            flat_dicts.append(flat_dict)

        return pd.DataFrame(flat_dicts)

    def pretty_print_summary(self) -> None:

        print("\n===== BENCHMARK SUMMARY =====\n")

        header = [
            "#RUN",
            "Lib",
            "Model",
            "Accelerator",
            "Precision",
            "BS",
            "it/s train",
            "it/s infer",
        ]
        table_data = []

        for i, result in enumerate(self.results):
            sw_framework = result.sw_info.ai_framework_name
            model = result.bench_info.model
            accelerator = result.hw_info.accelerator
            precision = result.bench_info.compute_precision
            train_throughput = round(result.train_performance.throughput, 2)
            infer_throughput = round(result.infer_performance.throughput, 2)

            assert (
                result.bench_info.batch_size_inference
                == result.bench_info.batch_size_training
            )
            batch_size = result.bench_info.batch_size_training

            row_data = [
                str(i),
                sw_framework,
                model,
                accelerator,
                precision,
                batch_size,
                train_throughput,
                infer_throughput,
            ]
            table_data.append(row_data)

        print(tabulate(table_data, headers=header, tablefmt="pretty"))

    def export_to_csv(self, file_name: str) -> None:

        df = self.to_dataframe()
        df.to_csv(file_name, index=False)

    def export_to_excel(self, file_name: str) -> None:

        df = self.to_dataframe()
        df.to_excel(file_name, index=False)
