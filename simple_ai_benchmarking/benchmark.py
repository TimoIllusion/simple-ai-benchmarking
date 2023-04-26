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
    
    result_log = add_iterations_per_second(result_log)
    
    return result_log

def add_iterations_per_second(result: BenchmarkResult) -> BenchmarkResult:
    result.iterations_per_second_inference =  result.num_iterations_eval / result.eval_duration_s
    result.iterations_per_second_training = result.num_iterations_training / result.train_duration_s
    return result


def main():
    
    workloads = [
        MLPMixer(128), 
        EfficientNet(None, 64)
        ]
    
    result_logger = Logger(log_dir="")
    
    for workload in workloads:
        benchmark_result = benchmark(workload)
        result_logger.add_result(benchmark_result)
    
    result_logger.print_info()
    result_logger.save()
        

if __name__ == "__main__":
    main()