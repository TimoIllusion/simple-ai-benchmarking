from abc import abstractmethod, ABC
import platform
import multiprocessing
from typing import Tuple
import datetime

from loguru import logger

import numpy as np
import psutil
import cpuinfo

from simple_ai_benchmarking.results import (
    SWInfo,
    HWInfo,
    BenchInfo,
    PerformanceResult,
    BenchmarkResult,
)
from simple_ai_benchmarking.config import AIWorkloadBaseConfig, AIModelWrapper
from simple_ai_benchmarking.dataset import Dataset


class AIWorkload(ABC):

    def __init__(
        self, ai_model: AIModelWrapper, dataset: Dataset, config: AIWorkloadBaseConfig
    ) -> None:

        self.model_name = ai_model.name
        self.model = ai_model.model

        self.cfg = config

        self.dataset = dataset

        self.reset_iteration_counter()

        self.warmup_done = False

    def reset_iteration_counter(self) -> None:
        self.iteration_counter = 0

    def _increment_iteration_counter_by_batch_size(self) -> None:
        self.iteration_counter += self.cfg.batch_size

    @abstractmethod
    def setup(self) -> None:
        pass

    def warmup(self) -> None:
        self._warmup()
        self.reset_iteration_counter()
        self.warmup_done = True
        self.warmup_inference_done = False

    @abstractmethod
    def _warmup(self) -> None:
        pass

    def execute(self) -> None:
        assert self.warmup_done, "Warmup not done before execution."
        self._execute()

    @abstractmethod
    def _execute(self) -> None:
        pass

    @abstractmethod
    def _get_ai_framework_version(self) -> str:
        pass

    @abstractmethod
    def _get_ai_framework_name(self) -> str:
        pass

    @abstractmethod
    def _get_ai_framework_extra_info(self) -> str:
        pass

    @abstractmethod
    def _get_accelerator_info(self) -> str:
        pass

    def build_result_log(self) -> BenchmarkResult:

        sw_info = SWInfo(
            ai_framework_name=self._get_ai_framework_name(),
            ai_framework_version=self._get_ai_framework_version(),
            ai_framework_extra_info=self._get_ai_framework_extra_info(),
            python_version=platform.python_version(),
            os_version=platform.platform(aliased=False, terse=False),
        )

        hw_info = HWInfo(
            cpu=cpuinfo.get_cpu_info()["brand_raw"],
            num_cores=multiprocessing.cpu_count(),
            ram_gb=psutil.virtual_memory().total / 1e9,
            accelerator=self._get_accelerator_info(),
        )

        bench_info = BenchInfo(
            workload_type=self.__class__.__name__,
            model=self.model_name,
            compute_precision=self.cfg.data_type.name,
            batch_size_training=self.cfg.batch_size,
            batch_size_inference=self.cfg.batch_size,
            sample_shape=None,
            date=datetime.datetime.now().isoformat(),
        )

        train_performance = PerformanceResult(iterations=self.iteration_counter)

        infer_performance = PerformanceResult(iterations=self.iteration_counter)

        benchmark_result = BenchmarkResult(
            sw_info=sw_info,
            hw_info=hw_info,
            bench_info=bench_info,
            train_performance=train_performance,
            infer_performance=infer_performance,
        )

        return benchmark_result

    def __str__(self) -> str:
        return f"{self.__class__.__name__} | {self.model_name} | {self._get_accelerator_info()}"
