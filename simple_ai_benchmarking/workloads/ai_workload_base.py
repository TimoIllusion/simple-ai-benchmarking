from abc import abstractmethod, ABC
import platform
import multiprocessing

import psutil

from simple_ai_benchmarking.log import *
from simple_ai_benchmarking.definitions import NumericalPrecision

class AIWorkloadBase(ABC):
    
    def __init__(self, model, epochs: int, num_batches: int, batch_size: int, device_name: str, data_type: NumericalPrecision):
        self.model = model
        self.epochs = epochs
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.device_name = device_name
        self.data_type = data_type

    @abstractmethod
    def setup(self) -> None:
        pass    
    
    @abstractmethod
    def train(self) -> None:
        pass
    
    @abstractmethod
    def eval(self) -> None:
        pass    
    
    @abstractmethod
    def infer(self) -> None:
        pass
    
    @abstractmethod
    def _get_ai_framework_version(self) -> str:
        pass
    
    @abstractmethod
    def _get_ai_framework_name(self) -> str:
        pass
    
    @abstractmethod
    def _get_accelerator_info(self) -> str:
        pass
    
    def build_result_log(self) -> BenchmarkResult:
        
        # SWInfo
        sw_info = SWInfo(
            ai_framework=self._get_ai_framework_name(),
            ai_framework_version=self._get_ai_framework_version(),
            python_version=platform.python_version()
        )
        
        hw_info = HWInfo(
            cpu=str(platform.processor()) + str(platform.architecture()),
            num_cores=multiprocessing.cpu_count(),
            ram_gb=psutil.virtual_memory().total / 1e9,  # You'd need to fill this in
            accelerator=self._get_accelerator_info()
        )
        
        # BenchInfo
        bench_info = BenchInfo(
            workload_type=self.__class__.__name__,
            compute_precision=self.data_type.name,
            batch_size_training=self.batch_size,
            batch_size_inference=self.batch_size,
            sample_shape=None  # You'd need to fill this in
        )
        
        # PerformanceResult (Note: These are placeholders; actual values should be filled later)
        train_performance = PerformanceResult(
            iterations=self.num_batches * self.batch_size * self.epochs
        )
        
        infer_performance = PerformanceResult(
            iterations=self.num_batches * self.batch_size
        )
        
        # Combine into BenchmarkResult
        benchmark_result = BenchmarkResult(
            sw_info=sw_info,
            hw_info=hw_info,
            bench_info=bench_info,
            train_performance=train_performance,
            infer_performance=infer_performance
        )
        
        return benchmark_result









    