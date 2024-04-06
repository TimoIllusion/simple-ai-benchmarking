# Note: This module defines the default configurations for the pytorch benchmark. By design, this is hardcoded and NOT easily customizable by the user,
# so that the user does not have to worry about the benchmark configuration. However, this may change in the future, since it is not very good programming style.

from typing import List
from copy import copy
from sys import platform

import torch
import torchvision
from loguru import logger

from simple_ai_benchmarking.definitions import (
    NumericalPrecision,
    AIWorkloadBaseConfig,
    AIModelWrapper,
)
from simple_ai_benchmarking.workloads.ai_workload import AIWorkload
from simple_ai_benchmarking.workloads.pytorch_workload import PyTorchTraining, PyTorchInference
from simple_ai_benchmarking.models.pt.simple_classification_cnn import (
    SimpleClassificationCNN,
)
from simple_ai_benchmarking.dataset import SyntheticPytorchDataset, DatasetConfig


# TODO: use workload factory to create workloads
def build_default_pt_workloads() -> List[AIWorkload]:

    device_name = get_device_name()

    logger.trace("Device Name: {}", device_name)

    model_shape = [3, 224, 224]
    
    dataset_config_bs8 = DatasetConfig(
        num_batches=50,
        batch_size=8,
        input_shape_without_batch=model_shape,
        target_shape_without_batch=[]
    )
    
    syn_dataset_bs8 = SyntheticPytorchDataset(dataset_config_bs8)
    syn_dataset_bs8.prepare()
    
    
    dataset_config_bs1 = copy(dataset_config_bs8)
    dataset_config_bs1.batch_size = 1
    
    syn_dataset_bs1 = SyntheticPytorchDataset(dataset_config_bs1)
    syn_dataset_bs1.prepare()

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

    simple_classification_cnn = AIModelWrapper(
        "SimpleClassificationCNN", SimpleClassificationCNN(100, model_shape)
    )
    resnet50 = AIModelWrapper("ResNet50", torchvision.models.resnet50(num_classes=1000))
    vitb16 = AIModelWrapper("ViT-B-16", torchvision.models.vit_b_16(num_classes=1000))

    workloads = [
        PyTorchInference(simple_classification_cnn, syn_dataset_bs8, common_cfg_default),
        PyTorchInference(simple_classification_cnn, syn_dataset_bs8, common_cfg_fp16_mixed),
        PyTorchInference(resnet50, syn_dataset_bs1, common_cfg_bs1_default),
        PyTorchInference(vitb16, syn_dataset_bs1, common_cfg_bs1_default),
        PyTorchTraining(simple_classification_cnn, syn_dataset_bs8, common_cfg_default),
        PyTorchTraining(simple_classification_cnn, syn_dataset_bs8, common_cfg_fp16_mixed),
        PyTorchTraining(resnet50, syn_dataset_bs1,common_cfg_bs1_default),
        PyTorchTraining(vitb16, syn_dataset_bs1, common_cfg_bs1_default),
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
