from simple_ai_benchmarking import AIWorkload
from simple_ai_benchmarking.workloads.mlpmixer import MLPMixer
from simple_ai_benchmarking.workloads.efficientnet import EfficientNet
from simple_ai_benchmarking.log import BenchmarkResult, Logger
from simple_ai_benchmarking.timer import Timer

def benchmark(workload: AIWorkload) -> BenchmarkResult:
    
    workload.setup()
     
    with Timer() as t:
        workload.train()
    training_duration_s = t.duration_s
    
    with Timer() as t:
        workload.eval()
    eval_duration_s = t.duration_s

    result_log = workload.build_result_log()
    
    result_log.train_duration_s = training_duration_s
    result_log.eval_duration_s = eval_duration_s
    
    result_log = calculate_iterations_per_second(result_log)
    
    return result_log

def calculate_iterations_per_second(result: BenchmarkResult):
    result.iterations_per_second_inference = result.eval_duration_s / result.num_iterations_eval
    result.iterations_per_second_training = result.train_duration_s / result.num_iterations_training
    return result


def main():
    
    workloads = [
        MLPMixer(128), 
        EfficientNet(64)
        ]
    
    result_logger = Logger(log_dir="")
    
    for workload in workloads:
        benchmark_result = benchmark(workload)
        result_logger.add_result(benchmark_result)
    
    result_logger.print_info()
    result_logger.save()
        

if __name__ == "__main__":
    main()