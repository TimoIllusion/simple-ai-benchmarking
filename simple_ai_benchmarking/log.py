import os
import json
from dataclasses import dataclass
from typing import List


@dataclass
class BenchmarkResult:
    sw_framework: str
    devices: str
    compute_precision: str
    batch_size_training: int
    num_iterations_training: int
    batch_size_eval: int
    num_iterations_eval: int
    sample_input_shape: list
    train_duration_s: float
    eval_duration_s: float
    iterations_per_second_training: float
    iterations_per_second_inference: float
    
    
class Logger:
    results: List[BenchmarkResult] = []
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        
    def add_result(self, result):
        self.results.append(result)
        
    def save(self):
        
        data_to_dump = [x.__dict__  for x in self.results]
        
        target_json_path = os.path.join(self.log_dir, "log.json")
        with open(target_json_path, "w") as f:
            json.dump(data_to_dump, f, indent=4)
        
        print("Saved logs to ", target_json_path)
    