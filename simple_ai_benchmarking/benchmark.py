from typing import List
import math

from loguru import logger

from simple_ai_benchmarking.workloads.ai_workload import AIWorkload
from simple_ai_benchmarking.results import BenchmarkLogger, BenchmarkResult
from simple_ai_benchmarking.timer import Timer


def process_workloads(
    workloads: List[AIWorkload], out_file_base="benchmark_results", repetitions=3
) -> None:

    assert workloads, "Got empty list fo workloads."

    result_logger = BenchmarkLogger()

    for workload in workloads:
        logger.info(f"Running benchmark: {workload}")
        benchmark_repetition_results = _repeat_benchmark_n_times(workload, repetitions)
        result_logger.add_repetitions_for_one_benchmark(benchmark_repetition_results)

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
    
    TARGET_BENCHMARK_DURATION_SECONDS = 0.1

    workload.setup()

    logger.info("WARMUP")
    workload.warmup()
    
    logger.info("CALIBRATION")
    with Timer() as t:
        workload.execute()
        
    train_calib_duration_s = t.duration_s
    train_repetitions = calculate_repetitions(TARGET_BENCHMARK_DURATION_SECONDS, train_calib_duration_s)

    workload.reset_iteration_counter()

    logger.info(f"{train_repetitions} EXECUTIONS of {workload.__class__.__name__}")
    with Timer() as t:
        for i in range(train_repetitions):
            logger.info(f"EXECUTION {i+1}/{train_repetitions}")
            workload.execute()
    training_duration_s = t.duration_s
    
    result_log = workload.build_result_log()

    result_log.update_train_performance_duration(training_duration_s)
    result_log.update_infer_performance_duration(training_duration_s)

    return result_log

def calculate_repetitions(target_duration_seconds: float, actual_duration_seconds: float) -> int:
    repetitions = math.ceil(target_duration_seconds / actual_duration_seconds)

    if repetitions == 0:
        repetitions = 1

    return int(repetitions)
