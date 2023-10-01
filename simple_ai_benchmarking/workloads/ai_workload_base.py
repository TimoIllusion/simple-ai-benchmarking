from abc import abstractmethod, ABC
import platform
import multiprocessing
from typing import Tuple

from loguru import logger
import numpy as np
import psutil

from simple_ai_benchmarking.log import *
from simple_ai_benchmarking.definitions import AIWorkloadBaseConfig

class AIWorkloadBase(ABC):
    
    def __init__(self, model, config: AIWorkloadBaseConfig):
        
        self.model = model
        self.cfg = config
        
        self.dataset_inputs_shape = [self.cfg.num_batches * self.cfg.batch_size] + list(self.cfg.input_shape_without_batch)
        self.dataset_targets_shape = [self.cfg.num_batches * self.cfg.batch_size] + list(self.cfg.target_shape_without_batch)
        
    def _generate_random_dataset_with_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        inputs = np.random.random(self.dataset_inputs_shape).astype(np.float32)
        targets = np.random.randint(low=0, high=2, size=self.dataset_targets_shape).astype(np.int64)
        
        logger.info("Synthetic Dataset NumPy Inputs Shape: {} {}", inputs.shape, inputs.dtype)
        logger.info("Synthetic Dataset NumPy Targets Shape: {} {}", targets.shape, targets.dtype)
        
        return inputs, targets
        

    @abstractmethod
    def setup(self) -> None:
        pass    
    
    def warmup(self) -> None:
        self.train()
        self.infer()
    
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
        
        sw_info = SWInfo(
            ai_framework=self._get_ai_framework_name(),
            ai_framework_version=self._get_ai_framework_version(),
            python_version=platform.python_version()
        )
        
        hw_info = HWInfo(
            cpu=str(platform.processor()) + str(platform.architecture()),
            num_cores=multiprocessing.cpu_count(),
            ram_gb=psutil.virtual_memory().total / 1e9,
            accelerator=self._get_accelerator_info()
        )
        
        bench_info = BenchInfo(
            workload_type=self.__class__.__name__,
            model=self.model.__class__.__name__,
            compute_precision=self.cfg.data_type.name,
            batch_size_training=self.cfg.batch_size,
            batch_size_inference=self.cfg.batch_size,
            sample_shape=None
        )
        
        train_performance = PerformanceResult(
            iterations=self.cfg.num_batches * self.cfg.batch_size * self.cfg.epochs
        )
        
        infer_performance = PerformanceResult(
            iterations=self.cfg.num_batches * self.cfg.batch_size
        )
        
        benchmark_result = BenchmarkResult(
            sw_info=sw_info,
            hw_info=hw_info,
            bench_info=bench_info,
            train_performance=train_performance,
            infer_performance=infer_performance
        )
        
        return benchmark_result









    