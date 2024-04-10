# Project Name: simple-ai-benchmarking
# File Name: config_pt_tf.py
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

#TODO: write model builder function to generate easy configs
def build_default_pt_workload_configs(
    framework: AIFramework,
) -> List[AIWorkloadBaseConfig]:

    input_sample_shape = ImageShape(224, 224, 3)
    num_classes = 100
    training_epochs = 10
    num_batches = 50

    if framework is AIFramework.PYTORCH:
        device_name = get_device_name_pytorch()
    elif framework is AIFramework.TENSORFLOW:
        device_name = get_device_name_tensorflow()
    else:
        raise ValueError("Invalid framework")

    logger.trace("Device Name: {}", device_name)
    logger.trace("Input Sample Shape: {}", input_sample_shape)
    
    dataset_sample_shape = input_sample_shape.to_tuple_depending_on_framework(framework)

    workload_cfgs = [
        InferenceConfig(
            dataset_cfg=DatasetConfig(
                batch_size=1,
                input_shape_without_batch=dataset_sample_shape,
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
                input_shape_without_batch=dataset_sample_shape,
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
                input_shape_without_batch=dataset_sample_shape,
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
                input_shape_without_batch=dataset_sample_shape,
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
                input_shape_without_batch=dataset_sample_shape,
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
