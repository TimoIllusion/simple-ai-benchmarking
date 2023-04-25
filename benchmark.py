import time
import json

from simple_ai_benchmarking import AIWorkload
from simple_ai_benchmarking.mlpmixer import MLPMixer
from simple_ai_benchmarking.efficientnet import EfficientNet

def benchmark(workload: AIWorkload) -> dict:
        
    workload.setup()
    
    t0 = time.time()
    workload.train()
    t0_delta = time.time() - t0
    
    t1 = time.time()
    workload.eval()
    t1_delta = time.time() - t1
    
    log = workload.build_log_dict()
    
    print("Elapsed time for training:", t0_delta)
    print("Elapsed time for inference:", t1_delta)
    
    log["train_duration_s"] = t0_delta
    log["eval_duration_s"] = t1_delta
    
    log["iterations_per_second_training"] = log["num_iterations_training"] / log["train_duration_s"]
    log["iterations_per_second_inference"] = log["num_iterations_eval"] / log["eval_duration_s"]
    
    return log

def main():
    
    workloads = [
        MLPMixer(), 
        EfficientNet()
        ]
    
    logs = []
    
    for workload in workloads:
        
        log = benchmark(workload)
        logs.append(log)
    
    print("\n===== BENCHMARKS FINISHED =====")
    print("Benchmark results log:")
    
    for workload, log in zip(workloads, logs):
        workload_name = workload.__class__.__name__
        log["workload_type"] = workload_name
        
        print(f"{workload_name} results:")
        [print(f"{key}: {val}") for key, val in log.items()]
        print("")
    
    with open("log.json", "w") as f:
        json.dump(logs, f, indent=4)
        

if __name__ == "__main__":
    main()