from abc import abstractmethod, ABC
import platform
import multiprocessing
import datetime

from loguru import logger

import psutil
import cpuinfo

from simple_ai_benchmarking.results import (
    SWInfo,
    HWInfo,
    BenchInfo,
    PerformanceResult,
    BenchmarkResult,
)
from simple_ai_benchmarking.config_structures import AIWorkloadBaseConfig, AIStage


class AIWorkload(ABC):

    def __init__(self, config: AIWorkloadBaseConfig) -> None:
        self.cfg = config

        self.warmup_done = False

    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def _prepare_synthetic_dataset(self) -> object:
        pass

    def warmup(self) -> None:
        """Warm up workload before execution."""

        self._warmup()
        self.warmup_done = True

    @abstractmethod
    def _warmup(self) -> None:
        pass

    @abstractmethod
    def prepare_execution(self) -> None:
        pass

    def execute(self) -> None:
        assert self.warmup_done, "Warmup not done before execution."
        self._execute()

    @abstractmethod
    def _execute(self) -> None:
        pass

    @abstractmethod
    def _calculate_iterations(self) -> int:
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

    @abstractmethod
    def _get_ai_stage(self) -> AIStage:
        pass

    @abstractmethod
    def _get_model_parameters(self) -> int:
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
            model=self.cfg.model_cfg.model_identifier.value,
            compute_precision=self.cfg.precision.name,
            batch_size=self.cfg.dataset_cfg.batch_size,
            sample_shape=self.cfg.dataset_cfg.input_shape_without_batch,
            date=datetime.datetime.now().isoformat(),
            num_classes=self.cfg.model_cfg.num_classes,
            num_parameters=self._get_model_parameters(),
        )

        performance = PerformanceResult(iterations=self._calculate_iterations())

        benchmark_result = BenchmarkResult(
            sw_info=sw_info,
            hw_info=hw_info,
            bench_info=bench_info,
            performance=performance,
        )

        return benchmark_result

    def __str__(self) -> str:
        return f"{self.__class__.__name__} | {self.cfg.model_cfg.model_identifier.value} | {self._get_accelerator_info()}"
