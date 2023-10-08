import time
import os
import shutil
import pandas
import numpy as np
import pytest

from simple_ai_benchmarking.workloads.ai_workload_base import AIWorkloadBase
from simple_ai_benchmarking.definitions import AIWorkloadBaseConfig, NumericalPrecision
from simple_ai_benchmarking.log import BenchmarkResult
from simple_ai_benchmarking.benchmark import benchmark, proccess_workloads

_PER_FUNCTION_TIME_DELAY_S = 0.1

@pytest.fixture(scope='session', autouse=True)
def prepare():
    _clean_up_prior_test_results()

def _clean_up_prior_test_results(): 
    _safe_remove("benchmark_results_empty.csv")
    _safe_remove("benchmark_results_empty.xlsx")
    _safe_remove("benchmark_results_dummy.csv")
    _safe_remove("benchmark_results_dummy.xlsx")

def _safe_remove(filename):
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

class DummyWorkload(AIWorkloadBase):
    def setup(self) -> None:
        pass    
    
    def warmup(self) -> None:
        self.train()
        self.infer()
    
    def train(self) -> None:
        time.sleep(_PER_FUNCTION_TIME_DELAY_S)
    
    def eval(self) -> None:
        time.sleep(_PER_FUNCTION_TIME_DELAY_S)    
    
    def infer(self) -> None:
        time.sleep(_PER_FUNCTION_TIME_DELAY_S)
    
    def _get_ai_framework_version(self) -> str:
        return "0.0.0"
    
    def _get_ai_framework_name(self) -> str:
        return "dummy"
    
    def _get_accelerator_info(self) -> str:
        return "dummy_accelerator"
    
def _prepare_benchmark_dummy_cfg() -> AIWorkloadBaseConfig:
    cfg = AIWorkloadBaseConfig(
        epochs=1,
        batch_size=1,
        num_batches=1,
        device_name="cpu",
        data_type=NumericalPrecision.DEFAULT_PRECISION,
        input_shape_without_batch=[10, 10, 10],
        target_shape_without_batch=[10, 10, 10],
    )
    return cfg

def test_benchmark_result_type():
    
    cfg = _prepare_benchmark_dummy_cfg()
    
    workload = DummyWorkload(None, cfg)
    result = benchmark(workload)
    
    assert isinstance(result, BenchmarkResult), f"Wrong type of benchmark result: {type(result)}"

def test_benchmark_timing():

    cfg = _prepare_benchmark_dummy_cfg()
    
    workload = DummyWorkload(None, cfg)
    
    t0 = time.time()
    _ = benchmark(workload)
    t1 = time.time()

    duration_s = t1 - t0

    target_duration_s = 4 * _PER_FUNCTION_TIME_DELAY_S
    tolerance_s = target_duration_s * 0.3

    assert np.fabs(duration_s - target_duration_s) <= tolerance_s, f"Benchmark duration differs from target duration. Target is {target_duration_s} but was {duration_s}."

def test_process_workloads_correct_num_workloads_output():

    cfg = _prepare_benchmark_dummy_cfg()
    
    workloads = [DummyWorkload(None, cfg), DummyWorkload(None, cfg), DummyWorkload(None, cfg)]
    proccess_workloads(workloads, "benchmark_results_dummy")
    
    assert os.path.exists("benchmark_results_dummy.csv"), "Result file does not exist"
    
    df = pandas.read_csv("benchmark_results_dummy.csv")
    assert len(df) == len(workloads), f"Results csv has wrong number of results, has {len(df)} but should have {len(workloads)}"

def test_process_workloads_empty_workloads():

    workloads = []
    proccess_workloads(workloads, "benchmark_results_empty")

    assert not os.path.exists("benchmark_results_empty.csv"), "Result file does exist, although workloads were empty."

