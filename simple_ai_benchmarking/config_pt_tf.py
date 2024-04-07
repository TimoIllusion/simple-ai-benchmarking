# Note: This module defines the default configurations for the pytorch benchmark. By design, this is hardcoded and NOT easily customizable by the user,
# so that the user does not have to worry about the benchmark configuration. However, this may change in the future, since it is not very good programming style.

from typing import List
from sys import platform

from loguru import logger

from simple_ai_benchmarking.config_structures import (
    NumericalPrecision,
    AIWorkloadBaseConfig,
    InferenceConfig,
    TrainingConfig,
    DatasetConfig,
    ImageShape,
    AIFramework,
    ClassificiationModelConfig,
    ModelIdentifier,
)


def build_default_pt_workload_configs(
    framework: AIFramework,
) -> List[AIWorkloadBaseConfig]:

    img_shape = ImageShape(224, 224, 3)
    num_classes = 100
    training_epochs = 10
    num_batches = 50

    if framework is AIFramework.PYTORCH:
        device_name = get_device_name_pytorch()
        input_sample_shape = img_shape.to_tuple_chw()
    elif framework is AIFramework.TENSORFLOW:
        device_name = get_device_name_tensorflow()
        input_sample_shape = img_shape.to_tuple_hwc()
    else:
        raise ValueError("Invalid framework")

    logger.trace("Device Name: {}", device_name)
    logger.trace("Input Sample Shape: {}", input_sample_shape)

    workload_cfgs = [
        InferenceConfig(
            dataset_cfg=DatasetConfig(
                batch_size=1,
                input_shape_without_batch=input_sample_shape,
                num_batches=num_batches,
            ),
            model_cfg=ClassificiationModelConfig(
                model_identifier=ModelIdentifier.VIT_B_16,
                model_shape=input_sample_shape,
                num_classes=num_classes,
            ),
            device_name=device_name,
            precision=NumericalPrecision.DEFAULT_PRECISION,
        ),
        InferenceConfig(
            dataset_cfg=DatasetConfig(
                batch_size=8,
                input_shape_without_batch=input_sample_shape,
                num_batches=num_batches,
            ),
            model_cfg=ClassificiationModelConfig(
                model_identifier=ModelIdentifier.SIMPLE_CLASSIFICATION_CNN,
                model_shape=input_sample_shape,
                num_classes=num_classes,
            ),
            device_name=device_name,
            precision=NumericalPrecision.DEFAULT_PRECISION,
        ),
        InferenceConfig(
            dataset_cfg=DatasetConfig(
                batch_size=8,
                input_shape_without_batch=input_sample_shape,
                num_batches=num_batches,
            ),
            model_cfg=ClassificiationModelConfig(
                model_identifier=ModelIdentifier.SIMPLE_CLASSIFICATION_CNN,
                model_shape=input_sample_shape,
                num_classes=num_classes,
            ),
            device_name=device_name,
            precision=NumericalPrecision.MIXED_FP16,
        ),
        TrainingConfig(
            dataset_cfg=DatasetConfig(
                batch_size=8,
                input_shape_without_batch=input_sample_shape,
                num_batches=num_batches,
            ),
            model_cfg=ClassificiationModelConfig(
                model_identifier=ModelIdentifier.SIMPLE_CLASSIFICATION_CNN,
                model_shape=input_sample_shape,
                num_classes=num_classes,
            ),
            device_name=device_name,
            precision=NumericalPrecision.DEFAULT_PRECISION,
            epochs=training_epochs,
        ),
        TrainingConfig(
            dataset_cfg=DatasetConfig(
                batch_size=8,
                input_shape_without_batch=input_sample_shape,
                num_batches=num_batches,
            ),
            model_cfg=ClassificiationModelConfig(
                model_identifier=ModelIdentifier.SIMPLE_CLASSIFICATION_CNN,
                model_shape=input_sample_shape,
                num_classes=num_classes,
            ),
            device_name=device_name,
            precision=NumericalPrecision.MIXED_FP16,
            epochs=training_epochs,
        ),
    ]

    return workload_cfgs


def get_device_name_pytorch() -> str:
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


def get_device_name_tensorflow() -> str:
    return "/gpu:0"
