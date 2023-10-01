from typing import List

from loguru import logger

from simple_ai_benchmarking.workloads.ai_workload_base import AIWorkloadBase
from simple_ai_benchmarking.log import *
from simple_ai_benchmarking.timer import Timer
from simple_ai_benchmarking.config import build_default_tf_workloads, build_default_pt_workloads

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

def _proccess_workloads(workloads: List[AIWorkloadBase], out_file_base="benchmark_results") -> List[BenchmarkResult]:
    result_logger = BenchmarkLogger()
    
    for workload in workloads:
        benchmark_result = benchmark(workload)
        result_logger.add_result(benchmark_result)
    
    result_logger.pretty_print_summary()
    
    result_logger.export_to_csv(out_file_base + ".csv")
    try:
        result_logger.export_to_excel(out_file_base + ".xlsx")
    except ModuleNotFoundError as e:
        logger.info("Could not export to excel:", e, "\nPlease install openpyxl to export to excel, e.g. via SAI [xlsx] extra.")
    
def run_tf_benchmarks():
    
    workloads = build_default_tf_workloads()
    _proccess_workloads(workloads, "benchmark_results_tf")
    
def run_pt_benchmarks():
    
    workloads = build_default_pt_workloads()
    _proccess_workloads(workloads, "benchmark_results_pt")
    

    
