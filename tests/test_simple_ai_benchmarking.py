import time
import os

import pytest

import pandas

from simple_ai_benchmarking.workloads.ai_workload import AIWorkload
from simple_ai_benchmarking.config_structures import (
    AIWorkloadBaseConfig,
    NumericalPrecision,
    AIModelWrapper,
)
from simple_ai_benchmarking.results import BenchmarkResult
from simple_ai_benchmarking.benchmark import benchmark, process_workloads

_PER_FUNCTION_TIME_DELAY_S = 0.1


@pytest.fixture(scope="session", autouse=True)
def prepare() -> None:
    _clean_up_prior_test_results()


def _clean_up_prior_test_results() -> None:

    _safe_remove("benchmark_results_dummy.csv")
    _safe_remove("benchmark_results_dummy.xlsx")


def _safe_remove(filename) -> None:
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass


class DummyWorkload(AIWorkload):

    def setup(self) -> None:
        pass

    def _warmup(self) -> None:
        self._execute()

    def _execute(self) -> None:
        time.sleep(_PER_FUNCTION_TIME_DELAY_S)

    def _get_ai_framework_version(self) -> str:
        return "0.0.0"

    def _get_ai_framework_name(self) -> str:
        return "dummy"

    def _get_accelerator_info(self) -> str:
        return "dummy_accelerator"

    def _get_ai_framework_extra_info(self) -> str:
        return "extra99.9"


def _prepare_benchmark_dummy_cfg() -> AIWorkloadBaseConfig:

    cfg = AIWorkloadBaseConfig(
        epochs=1,
        batch_size=1,
        num_batches=1,
        device_name="cpu",
        precision=NumericalPrecision.DEFAULT_PRECISION,
        input_shape_without_batch_hwc=[10, 10, 10],
        target_shape_without_batch_hwc=[10, 10, 10],
    )

    return cfg


def test_benchmark_result_type() -> None:

    cfg = _prepare_benchmark_dummy_cfg()

    dummy = AIModelWrapper("dummy", None)

    workload = DummyWorkload(dummy, cfg)
    result = benchmark(workload)

    assert isinstance(
        result, BenchmarkResult
    ), f"Wrong type of benchmark result: {type(result)}"


def test_process_workloads_correct_num_workloads_output() -> None:

    cfg = _prepare_benchmark_dummy_cfg()

    dummy = AIModelWrapper("dummy", None)
    workloads = [
        DummyWorkload(dummy, cfg),
        DummyWorkload(dummy, cfg),
        DummyWorkload(dummy, cfg),
    ]
    process_workloads(workloads, "benchmark_results_dummy")

    assert os.path.exists("benchmark_results_dummy.csv"), "Result file does not exist"

    df = pandas.read_csv("benchmark_results_dummy.csv")
    assert len(df) == len(
        workloads
    ), f"Results csv has wrong number of results, has {len(df)} but should have {len(workloads)}"
