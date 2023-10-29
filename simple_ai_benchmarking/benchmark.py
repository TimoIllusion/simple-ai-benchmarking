from typing import List

from loguru import logger

from simple_ai_benchmarking.workloads.ai_workload_base import AIWorkloadBase
from simple_ai_benchmarking.log import *
from simple_ai_benchmarking.timer import Timer
from simple_ai_benchmarking.config import build_default_tf_workloads, build_default_pt_workloads

def run_tf_benchmarks():
    
    workloads = build_default_tf_workloads()
    proccess_workloads(workloads, "benchmark_results_tf")
    
def run_pt_benchmarks():
    
    workloads = build_default_pt_workloads()
    proccess_workloads(workloads, "benchmark_results_pt")

def proccess_workloads(workloads: List[AIWorkloadBase], out_file_base="benchmark_results", repetitions=3):

    if not workloads:
        logger.info(f"Got empty list fo workloads: {workloads} -> Please check config.py to set workload configuration.")
        return []

    result_logger = BenchmarkLogger()
    
    for workload in workloads:
        benchmark_repetition_results = _repeat_benchmark_n_times(workload, repetitions)
        result_logger.add_repetitions_for_one_benchmark(benchmark_repetition_results)
    
    result_logger.pretty_print_summary()
    
    result_logger.export_to_csv(out_file_base + ".csv")
    try:
        result_logger.export_to_excel(out_file_base + ".xlsx")
    except ModuleNotFoundError as e:
        logger.warning(f"Could not export to excel: \"{e}\" -> Please install openpyxl to export to excel, e.g. via SAI [xlsx] extra.")

def _repeat_benchmark_n_times(workload: AIWorkloadBase, n_repetitions: int) -> List[BenchmarkResult]:
    benchmark_repetition_results = []
    for i in range(n_repetitions):
        logger.info(f"Repetition ({i+1}/{n_repetitions})")
        benchmark_result = benchmark(workload)
        benchmark_repetition_results.append(benchmark_result)

    return benchmark_repetition_results

def benchmark(workload: AIWorkloadBase) -> BenchmarkResult:
    
    workload.setup()
    
    logger.info("WARMUP")
    workload.warmup()
     
    with Timer() as t:
        logger.info("TRAINING")
        workload.train()
    training_duration_s = t.duration_s
    
    with Timer() as t:
        logger.info("INFERENCE")
        workload.infer()
    infer_duration_s = t.duration_s

    result_log = workload.build_result_log()
    
    result_log.update_train_performance_duration(training_duration_s)
    result_log.update_infer_performance_duration(infer_duration_s)
    
    return result_log


    

    
