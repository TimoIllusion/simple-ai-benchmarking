# Note: This module defines the default configurations for the pytorch benchmark. By design, this is hardcoded and NOT easily customizable by the user,
# so that the user does not have to worry about the benchmark configuration. However, this may change in the future, since it is not very good programming style.

from typing import List
from copy import copy
from sys import platform

import torch
import torchvision

from simple_ai_benchmarking.definitions import NumericalPrecision, AIWorkloadBaseConfig
from simple_ai_benchmarking.workloads.ai_workload import AIWorkload
from simple_ai_benchmarking.workloads.pytorch_workload import PyTorchWorkload
from simple_ai_benchmarking.models.pt.simple_classification_cnn import (
    PTSimpleClassificationCNN,
)


# TODO: use workload factory to create workloads
def build_default_pt_workloads() -> List[AIWorkload]:

    device_name = get_device_name()

    model_shape = [3, 224, 224]

    common_cfg_default = AIWorkloadBaseConfig(
        batch_size=8,
        num_batches=50,
        epochs=10,
        input_shape_without_batch=model_shape,
        target_shape_without_batch=[],
        device_name=device_name,
        data_type=NumericalPrecision.DEFAULT_PRECISION,
    )

    common_cfg_fp16_mixed = copy(common_cfg_default)
    common_cfg_fp16_mixed.data_type = NumericalPrecision.MIXED_FP16

    # common_cfg_fp32_explicit = copy(common_cfg_default)
    # common_cfg_fp32_explicit.data_type = NumericalPrecision.EXPLICIT_FP32

    common_cfg_bs1_default = copy(common_cfg_default)
    common_cfg_bs1_default.batch_size = 1

    workloads = [
        PyTorchWorkload(
            PTSimpleClassificationCNN(100, model_shape), common_cfg_default
        ),
        PyTorchWorkload(
            PTSimpleClassificationCNN(100, model_shape), common_cfg_fp16_mixed
        ),
        PyTorchWorkload(
            torchvision.models.resnet50(num_classes=1000), common_cfg_bs1_default
        ),
    ]

    return workloads


def get_device_name() -> str:

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
