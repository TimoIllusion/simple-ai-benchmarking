import time
import json

from simple_ai_benchmarking.mlpmixer import MLPMixer

def main():
    
    
    workload = MLPMixer()
    
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
    
    log["samples_per_second_training"] = log["num_samples_training"] / log["train_duration_s"]
    log["samples_per_second_inference"] = log["num_samples_eval"] / log["eval_duration_s"]
    
    
    print("Log info:", log)
    with open("log.json", "w") as f:
        json.dump(log, f, indent=4)
    

if __name__ == "__main__":
    main()