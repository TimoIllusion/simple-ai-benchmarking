# Project Name: simple-ai-benchmarking
# File Name: factory.py
# Author: Timo Leitritz
# Copyright (C) 2024 Timo Leitritz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


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
