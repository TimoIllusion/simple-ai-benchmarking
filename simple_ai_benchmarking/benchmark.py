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
        combined_benchmark_result = _repeat_benchmark_n_times(workload, repetitions)
        result_logger.add_result(combined_benchmark_result)
    
    result_logger.pretty_print_summary()
    
    result_logger.export_to_csv(out_file_base + ".csv")
    try:
        result_logger.export_to_excel(out_file_base + ".xlsx")
    except ModuleNotFoundError as e:
        logger.warning(f"Could not export to excel: \"{e}\" -> Please install openpyxl to export to excel, e.g. via SAI [xlsx] extra.")

def _repeat_benchmark_n_times(workload: AIWorkloadBase, n_repetitions: int) -> BenchmarkResult:
    benchmark_repetition_result = []
    for i in range(n_repetitions):
        logger.info(f"Repetition ({i+1}/{n_repetitions})")
        benchmark_result = benchmark(workload)
        benchmark_repetition_result.append(benchmark_result)
    combined_benchmark_result = _average_benchmark_results(benchmark_repetition_result)
    return combined_benchmark_result

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

def _average_benchmark_results(benchmark_results: List[BenchmarkResult]) -> BenchmarkResult:
        assert benchmark_results, "Got empty list of benchmark results"
    
        infer_performances = [result.infer_performance for result in benchmark_results]
        train_performances = [result.train_performance for result in benchmark_results]
        
        infer_avg_perf = _accumulate_and_average_performance_results(infer_performances)
        train_avg_perf = _accumulate_and_average_performance_results(train_performances)
        
        combined_avg_benchmark_result = benchmark_results[0]
        combined_avg_benchmark_result.infer_performance = infer_avg_perf
        combined_avg_benchmark_result.train_performance = train_avg_perf
        
        return combined_avg_benchmark_result
    
def _accumulate_and_average_performance_results(perf_results: List[PerformanceResult]) -> PerformanceResult:
    
    iterations_sum = 0
    duration_s_sum = 0.0
    
    for perf in perf_results:
        iterations_sum += perf.iterations
        duration_s_sum += perf.duration_s
        
    avg_result = PerformanceResult(iterations_sum)
    avg_result.update_duration_and_calc_throughput(duration_s_sum)
    
    return avg_result


    

    
