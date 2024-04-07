from simple_ai_benchmarking.workloads.ai_workload import AIWorkload
from simple_ai_benchmarking.config import (
    AIWorkloadBaseConfig,
    PytorchInferenceConfig,
    PytorchTrainingConfig,
    TensorflowInferenceConfig,
    TensorflowTrainingConfig,
)


class WorkloadFactory:

    @staticmethod
    def create_pytorch_workload(workload_cfg: AIWorkloadBaseConfig) -> AIWorkload:

        if type(workload_cfg) is PytorchInferenceConfig:
            from simple_ai_benchmarking.workloads.pytorch_workload import (
                PyTorchInference,
            )

            return PyTorchInference(workload_cfg)
        elif type(workload_cfg) is PytorchTrainingConfig:
            from simple_ai_benchmarking.workloads.pytorch_workload import (
                PyTorchTraining,
            )

            return PyTorchTraining(workload_cfg)
        else:
            raise ValueError(f"Workload type {type(workload_cfg)} not supported")

    @staticmethod
    def create_tensorflow_workload(workload_cfg: AIWorkloadBaseConfig) -> AIWorkload:
        if type(workload_cfg) is TensorflowInferenceConfig:
            from simple_ai_benchmarking.workloads.tensorflow_workload import (
                TensorFlowInference,
            )

            return TensorFlowInference(workload_cfg)
        elif type(workload_cfg) is TensorflowTrainingConfig:
            from simple_ai_benchmarking.workloads.tensorflow_workload import (
                TensorFlowTraining,
            )

            return TensorFlowTraining(workload_cfg)
        else:
            raise ValueError(f"Workload type {type(workload_cfg)} not supported")
