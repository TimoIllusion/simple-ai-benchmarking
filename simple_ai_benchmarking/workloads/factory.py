from typing import List

from simple_ai_benchmarking.workloads.ai_workload import AIWorkload
from simple_ai_benchmarking.config_structures import (
    AIWorkloadBaseConfig,
    InferenceConfig,
    TrainingConfig,
    AIFramework,
)


class WorkloadFactory:

    @staticmethod
    def create_workload(workload_cfg: AIWorkloadBaseConfig, framework: AIFramework) -> AIWorkload:
        if framework is AIFramework.PYTORCH:
            return WorkloadFactory._create_pytorch_workload(workload_cfg)
        elif framework is AIFramework.TENSORFLOW:
            return WorkloadFactory._create_tensorflow_workload(workload_cfg)
        else:
            raise ValueError(f"Framework {framework} not supported")

    @staticmethod
    def _create_pytorch_workload(workload_cfg: AIWorkloadBaseConfig) -> AIWorkload:

        if type(workload_cfg) is InferenceConfig:
            from simple_ai_benchmarking.workloads.pytorch_workload import (
                PyTorchInference,
            )

            return PyTorchInference(workload_cfg)
        elif type(workload_cfg) is TrainingConfig:
            from simple_ai_benchmarking.workloads.pytorch_workload import (
                PyTorchTraining,
            )

            return PyTorchTraining(workload_cfg)
        else:
            raise ValueError(f"Workload type {type(workload_cfg)} not supported")

    @staticmethod
    def _create_tensorflow_workload(workload_cfg: AIWorkloadBaseConfig) -> AIWorkload:
        if type(workload_cfg) is InferenceConfig:
            from simple_ai_benchmarking.workloads.tensorflow_workload import (
                TensorFlowInference,
            )

            return TensorFlowInference(workload_cfg)
        elif type(workload_cfg) is TrainingConfig:
            from simple_ai_benchmarking.workloads.tensorflow_workload import (
                TensorFlowTraining,
            )

            return TensorFlowTraining(workload_cfg)
        else:
            raise ValueError(f"Workload type {type(workload_cfg)} not supported")
    
    @staticmethod   
    def build_multiple_workloads(workload_configs: List[AIWorkloadBaseConfig], framework: AIFramework) -> List[AIWorkload]:
    
        workloads = []
        for cfg in workload_configs:
            workload = WorkloadFactory.create_workload(cfg, framework)
            workloads.append(workload)

        return workloads
