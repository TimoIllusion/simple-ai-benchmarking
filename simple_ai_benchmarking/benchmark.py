# Project Name: simple-ai-benchmarking
# File Name: benchmark.py
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


from typing import List
from multiprocessing import Process, Queue

from loguru import logger

from simple_ai_benchmarking.workloads.ai_workload import AIWorkload
from simple_ai_benchmarking.results import BenchmarkLogger, BenchmarkResult
from simple_ai_benchmarking.timer import Timer
from simple_ai_benchmarking.dataset import get_available_memory_in_bytes


def process_workloads(
    workloads: List[AIWorkload], out_file_base="benchmark_results", repetitions=3
) -> None:

    assert workloads, "Got empty list fo workloads."

    result_logger = BenchmarkLogger()

    for workload in workloads:
        logger.info(f"Running benchmark: {workload}")
        benchmark_repetition_results = _repeat_benchmark_n_times(workload, repetitions)
        result_logger.add_benchmark_result_by_averaging_multiple_results(
            benchmark_repetition_results
        )

    result_logger.pretty_print_summary()

    result_logger.export_to_csv(out_file_base + ".csv")
    try:
        result_logger.export_to_excel(out_file_base + ".xlsx")
    except ModuleNotFoundError as e:
        logger.warning(
            f'Could not export to excel: "{e}" -> Please install openpyxl to export to excel, e.g. via SAI [xlsx] extra.'
        )


def _repeat_benchmark_n_times(
    workload: AIWorkload, n_repetitions: int
) -> List[BenchmarkResult]:
    benchmark_repetition_results = []
    for i in range(n_repetitions):
        logger.info(f"Repetition ({i+1}/{n_repetitions})")
        result_queue = Queue()
        p = Process(target=_benchmark_process, args=(workload, result_queue))
        p.start()
        p.join()  # Wait for the process to complete
        benchmark_result = result_queue.get()  # Retrieve the result from the process
        benchmark_repetition_results.append(benchmark_result)

    return benchmark_repetition_results


def _benchmark_process(workload: AIWorkload, result_queue: Queue) -> None:
    benchmark_result = benchmark(workload)
    result_queue.put(benchmark_result)  # Send the result back to the parent process


def benchmark(workload: AIWorkload) -> BenchmarkResult:
    check_memory("before START")
    workload.setup()
    check_memory("after SETUP")

    logger.info(f"WARMUP: {workload.__class__.__name__}")
    workload.warmup()
    check_memory("after WARMUP")

    logger.info(f"EXECUTION: {workload.__class__.__name__}")
    workload.prepare_execution()
    check_memory("after EXECUTION PREPARATION")
    with Timer() as t:
        workload.execute()
    training_duration_s = t.duration_s

    result_log = workload.build_result_log()

    result_log.update_performance_duration(training_duration_s)

    return result_log


def check_memory(info_text: str = ""):
    available_memory_gb = get_available_memory_in_bytes() / 1e9
    logger.trace(f"Available memory {info_text} : {available_memory_gb} GB")
