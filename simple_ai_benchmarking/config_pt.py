# Note: This module defines the default configurations for the pytorch benchmark. By design, this is hardcoded and NOT easily customizable by the user,
# so that the user does not have to worry about the benchmark configuration. However, this may change in the future, since it is not very good programming style.

from typing import List
from sys import platform

from loguru import logger

from simple_ai_benchmarking.config import (
    NumericalPrecision,
    AIWorkloadBaseConfig,
    PytorchInferenceConfig,
    PytorchTrainingConfig,
    DatasetConfig,
    ImageShape,
)
from simple_ai_benchmarking.workloads.ai_workload import AIWorkload
from simple_ai_benchmarking.workloads.factory import WorkloadFactory


def build_default_pt_workloads() -> List[AIWorkload]:

    device_name = get_device_name()

    logger.trace("Device Name: {}", device_name)

    img_shape = ImageShape(224, 224, 3)
    input_sample_shape = img_shape.to_tuple_chw()

    workload_cfgs = [
        PytorchInferenceConfig(
            dataset_cfg=DatasetConfig(
                batch_size=8, input_shape_without_batch=input_sample_shape
            ),
            device_name=device_name,
        ),
        PytorchTrainingConfig(
            dataset_cfg=DatasetConfig(
                batch_size=8, input_shape_without_batch=input_sample_shape
            ),
            device_name=device_name,
        ),
        PytorchInferenceConfig(
            dataset_cfg=DatasetConfig(
                batch_size=8, input_shape_without_batch=input_sample_shape
            ),
            device_name=device_name,
            precision=NumericalPrecision.MIXED_FP16,
        ),
        PytorchTrainingConfig(
            dataset_cfg=DatasetConfig(
                batch_size=8, input_shape_without_batch=input_sample_shape
            ),
            device_name=device_name,
            precision=NumericalPrecision.MIXED_FP16,
        ),
    ]

    workloads = [WorkloadFactory.create_pytorch_workload(cfg) for cfg in workload_cfgs]

    return workloads


def get_device_name() -> str:
    import torch

    if platform == "linux" or platform == "linux2" or platform == "win32":

        if torch.cuda.is_available():
            device_name = "cuda"
        else:
            device_name = "cpu"

    elif platform == "darwin":

        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device_name = "mps"
        else:
            device_name = "cpu"

    else:
        device_name = "cpu"

    return device_name
