# Project Name: simple-ai-benchmarking
# File Name: results.py
# Author: Timo Leitritz
# Copyright (C) 2024 Timo Leitritz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from dataclasses import dataclass, field, asdict
from typing import List
import sys
import platform
import multiprocessing

from loguru import logger

import pandas as pd
import psutil
import cpuinfo
from tabulate import tabulate

from simple_ai_benchmarking.benchmark_metadata import (
    CV_BENCHMARK_FAMILY,
    CV_RUNNER_ID,
    CV_SPEC_NAME,
    SPEC_VERSION,
    assign_benchmark_identity,
    build_cv_profile,
    build_cv_profile_id,
    build_payload_hash,
)


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


def collect_sw_info(
    ai_framework_name: str,
    ai_framework_version: str,
    ai_framework_extra_info: str,
) -> SWInfo:
    """Gather host software info shared by every benchmark family."""
    return SWInfo(
        ai_framework_name=ai_framework_name,
        ai_framework_version=ai_framework_version,
        ai_framework_extra_info=ai_framework_extra_info,
        python_version=platform.python_version(),
        os_version=platform.platform(aliased=False, terse=False),
    )


def collect_hw_info(accelerator: str) -> HWInfo:
    """Gather host hardware info shared by every benchmark family."""
    return HWInfo(
        cpu=cpuinfo.get_cpu_info().get("brand_raw", "unknown"),
        num_cores=multiprocessing.cpu_count(),
        ram_gb=psutil.virtual_memory().total / 1e9,
        accelerator=accelerator,
    )


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
    batch_size: int
    date: str
    sample_shape: List[int]
    num_classes: int
    num_parameters: int
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
        profile = build_cv_profile(
            workload_type=self.workload_type,
            model=self.model,
            compute_precision=self.compute_precision,
            batch_size=self.batch_size,
            sample_shape=self.sample_shape,
            num_classes=self.num_classes,
        )
        assign_benchmark_identity(
            self,
            family=CV_BENCHMARK_FAMILY,
            spec_name=CV_SPEC_NAME,
            spec_version=SPEC_VERSION,
            runner_id=CV_RUNNER_ID,
            profile=profile,
            profile_id=build_cv_profile_id(profile),
        )


@dataclass
class BenchmarkResult:
    sw_info: SWInfo
    hw_info: HWInfo
    bench_info: BenchInfo
    performance: PerformanceResult

    def update_performance_duration(self, duration_s: float) -> None:
        self.performance.update_duration_and_calc_throughput(duration_s)


class BaseBenchmarkLogger:
    """Shared result collection and export for all benchmark families.

    Holds the family-agnostic machinery: collecting results, flattening nested
    dataclasses into a dataframe (with the per-row payload hash), and CSV/Excel
    export. Subclasses add family-specific result averaging and summary tables."""

    def __init__(self) -> None:
        self.results: list = []

    def add_result(self, result) -> None:
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

    def export_to_csv(self, file_name: str) -> None:
        self.to_dataframe().to_csv(file_name, index=False)

    def export_to_excel(self, file_name: str) -> None:
        self.to_dataframe().to_excel(file_name, index=False)

    def add_benchmark_result_by_averaging_multiple_results(self, results) -> None:
        """Average repetitions of one workload into a single result row.

        The non-performance fields are identical across repetitions, so the first
        result is reused and only its performance is replaced by the family-specific
        average (CV pools iterations/duration; LLM pools tokens/duration)."""
        assert results, "Got empty list of benchmark results"
        averaged_result = results[0]
        averaged_result.performance = self._average_performance(
            [result.performance for result in results]
        )
        self.add_result(averaged_result)

    def _average_performance(self, performances):
        raise NotImplementedError

    def pretty_print_summary(self) -> None:
        raise NotImplementedError


class BenchmarkLogger(BaseBenchmarkLogger):

    def __init__(self) -> None:
        self.results: List[BenchmarkResult] = []

    def _average_performance(
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

    def pretty_print_summary(self) -> None:

        print("\n===== BENCHMARK SUMMARY =====\n")

        header = [
            "#RUN",
            "WorkloadType",
            "Lib",
            "Model",
            "Accelerator",
            "Precision",
            "BS",
            "it/s",
        ]
        table_data = []

        for i, result in enumerate(self.results):
            workload_type = result.bench_info.workload_type
            sw_framework = result.sw_info.ai_framework_name
            model = result.bench_info.model
            accelerator = result.hw_info.accelerator
            precision = result.bench_info.compute_precision
            throughput = round(result.performance.throughput, 2)

            batch_size = result.bench_info.batch_size

            row_data = [
                str(i),
                workload_type,
                sw_framework,
                model,
                accelerator,
                precision,
                batch_size,
                throughput,
            ]
            table_data.append(row_data)

        print(tabulate(table_data, headers=header, tablefmt="pretty"))
