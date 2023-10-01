from dataclasses import dataclass, field, asdict
from typing import List

from loguru import logger

import pandas as pd
from tabulate import tabulate

@dataclass
class SWInfo:
    ai_framework: str
    ai_framework_version: str
    python_version: str

@dataclass
class HWInfo:
    cpu: str
    num_cores: int
    ram_gb: float
    accelerator: str

@dataclass
class PerformanceResult:
    iterations: int
    duration_s: float = field(init=False)
    throughput: float = field(init=False)
    finished_successfully: bool = True
    error_message: str = ""
    
    def update_duration_and_calc_throughput(self, duration_s: float) -> float:   
        self.duration_s = duration_s     
        self.throughput = self.iterations / self.duration_s if self.duration_s > 0 else 0.0

@dataclass
class BenchInfo:
    workload_type: str
    model: str
    compute_precision: str
    batch_size_training: int
    batch_size_inference: int
    sample_shape: List[int]

@dataclass
class BenchmarkResult:
    sw_info: SWInfo
    hw_info: HWInfo
    bench_info: BenchInfo
    train_performance: PerformanceResult
    infer_performance: PerformanceResult
    
    def update_train_performance_duration(self, duration_s: float):
        self.train_performance.update_duration_and_calc_throughput(duration_s)
        
    def update_infer_performance_duration(self, duration_s: float):
        self.infer_performance.update_duration_and_calc_throughput(duration_s)

class BenchmarkLogger:
    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def add_result(self, result: BenchmarkResult):
        self.results.append(result)

    def to_dataframe(self) -> pd.DataFrame:
        # Convert each BenchmarkResult to a nested dictionary
        nested_dicts = [asdict(result) for result in self.results]
        
        # Flatten the nested dictionaries for Pandas
        flat_dicts = []
        for nested_dict in nested_dicts:
            flat_dict = {}
            for key, value in nested_dict.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flat_dict[f"{key}_{sub_key}"] = sub_value
                else:
                    flat_dict[key] = value
            flat_dicts.append(flat_dict)
        
        return pd.DataFrame(flat_dicts)

    def pretty_print_summary(self):
        print("\n===== BENCHMARK SUMMARY =====\n")
        
        header = ["#RUN", "Lib", "Model", "Accelerator", "Precision", "BS", "it/s train", "it/s infer"]
        table_data = []

        for i, result in enumerate(self.results):
            sw_framework = result.sw_info.ai_framework
            model = result.bench_info.model
            accelerator = result.hw_info.accelerator
            precision = result.bench_info.compute_precision
            train_throughput = round(result.train_performance.throughput, 2)
            infer_throughput = round(result.infer_performance.throughput, 2)
            
            assert result.bench_info.batch_size_inference == result.bench_info.batch_size_training
            batch_size = result.bench_info.batch_size_training

            row_data = [str(i), sw_framework, model, accelerator, precision, batch_size, train_throughput, infer_throughput]
            table_data.append(row_data)

        print(tabulate(table_data, headers=header, tablefmt="pretty"))

    def export_to_csv(self, file_name: str):
        df = self.to_dataframe()
        df.to_csv(file_name, index=False)
        
    def export_to_excel(self, file_name: str):
        df = self.to_dataframe()
        df.to_excel(file_name, index=False)