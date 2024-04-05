from abc import abstractmethod, ABC
import platform
import multiprocessing
from typing import Tuple
import datetime

from loguru import logger

import numpy as np
import psutil
import cpuinfo

from simple_ai_benchmarking.log import (
    SWInfo,
    HWInfo,
    BenchInfo,
    PerformanceResult,
    BenchmarkResult,
)
from simple_ai_benchmarking.definitions import AIWorkloadBaseConfig, AIModelWrapper


class AIWorkload(ABC):

    def __init__(self, ai_model: AIModelWrapper, config: AIWorkloadBaseConfig) -> None:

        self.model_name = ai_model.name
        self.model = ai_model.model

        self.cfg = config

        self.reset_train_and_infer_iteration_counters()

        self.warmup_done = False

    def reset_train_and_infer_iteration_counters(self) -> None:
        self.infer_iteration_counter = 0
        self.train_iteration_counter = 0

    def _increment_infer_iteration_counter_by_batch_size(self) -> None:
        self.infer_iteration_counter += self.cfg.batch_size

    def _increment_train_iteration_counter_by_batch_size(self) -> None:
        self.train_iteration_counter += self.cfg.batch_size

    def _generate_random_dataset_with_numpy(self) -> Tuple[np.ndarray, np.ndarray]:

        self.dataset_inputs_shape = [self.cfg.num_batches * self.cfg.batch_size] + list(
            self.cfg.input_shape_without_batch
        )
        self.dataset_targets_shape = [
            self.cfg.num_batches * self.cfg.batch_size
        ] + list(self.cfg.target_shape_without_batch)

        inputs = np.random.random(self.dataset_inputs_shape).astype(np.float32)
        targets = np.random.randint(
            low=0, high=2, size=self.dataset_targets_shape
        ).astype(np.int64)

        logger.debug(
            "Synthetic Dataset NumPy Inputs Shape: {} {}", inputs.shape, inputs.dtype
        )
        logger.debug(
            "Synthetic Dataset NumPy Targets Shape: {} {}", targets.shape, targets.dtype
        )

        return inputs, targets

    @abstractmethod
    def setup(self) -> None:
        pass

    def warmup(self) -> None:
        self._warmup()
        self.reset_train_and_infer_iteration_counters()
        self.warmup_done = True

    @abstractmethod
    def _warmup(self) -> None:
        pass

    def train(self) -> None:
        assert self.warmup_done, "Warmup not done before training."
        self._train()

    @abstractmethod
    def _train(self) -> None:
        pass

    @abstractmethod
    def eval(self) -> None:
        pass

    def infer(self) -> None:
        assert self.warmup_done, "Warmup not done before inference."
        self._infer()

    @abstractmethod
    def _infer(self) -> None:
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

        train_performance = PerformanceResult(
            iterations=self.train_iteration_counter
        )

        infer_performance = PerformanceResult(
            iterations=self.infer_iteration_counter
        )

        benchmark_result = BenchmarkResult(
            sw_info=sw_info,
            hw_info=hw_info,
            bench_info=bench_info,
            train_performance=train_performance,
            infer_performance=infer_performance,
        )

        return benchmark_result

    def __str__(self) -> str:
        return str(self.model_name) + " on " + str(self._get_accelerator_info())
