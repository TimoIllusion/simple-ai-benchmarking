from typing import List
import time

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
        benchmark_result = benchmark(workload)
        benchmark_repetition_results.append(benchmark_result)

    return benchmark_repetition_results


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
    logger.warning(f"Available memory {info_text} : {available_memory_gb} GB")
