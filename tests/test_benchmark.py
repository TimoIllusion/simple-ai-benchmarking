import time
import os

from simple_ai_benchmarking.workloads.ai_workload_base import AIWorkloadBase
from simple_ai_benchmarking.definitions import AIWorkloadBaseConfig, NumericalPrecision
from simple_ai_benchmarking.benchmark import _proccess_workloads


class DummyWorkload(AIWorkloadBase):
    def setup(self) -> None:
        pass    
    
    def warmup(self) -> None:
        self.train()
        self.infer()
    
    def train(self) -> None:
        time.sleep(1)
    
    def eval(self) -> None:
        time.sleep(1)    
    
    def infer(self) -> None:
        time.sleep(1)
    
    def _get_ai_framework_version(self) -> str:
        return "0.0.0"
    
    def _get_ai_framework_name(self) -> str:
        return "dummy"
    
    def _get_accelerator_info(self) -> str:
        return "dummy_accelerator"
    

def test_benchmark_dummy():
    
    cfg = AIWorkloadBaseConfig(
        epochs=1,
        batch_size=1,
        num_batches=1,
        device_name="cpu",
        data_type=NumericalPrecision.DEFAULT_PRECISION,
        input_shape_without_batch=[10, 10, 10],
        target_shape_without_batch=[10, 10, 10],
    )
    
    workloads = [DummyWorkload(None, cfg)]
    _proccess_workloads(workloads, "benchmark_results_dummy")
    
    assert os.path.exists("benchmark_results_dummy.csv")