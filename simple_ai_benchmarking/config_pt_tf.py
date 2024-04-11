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


from typing import List, Tuple, Sequence
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

    INPUT_SAMPLE_SHAPE = ImageShape(224, 224, 3)

    if framework is AIFramework.PYTORCH:
        device_name = get_device_name_pytorch()
    elif framework is AIFramework.TENSORFLOW:
        device_name = get_device_name_tensorflow()
    else:
        raise ValueError("Invalid framework")

    logger.trace("Device Name: {}", device_name)
    logger.trace("Input Sample Shape: {}", INPUT_SAMPLE_SHAPE)

    dataset_sample_shape = INPUT_SAMPLE_SHAPE.to_tuple_depending_on_framework(framework)

    workload_cfgs = create_standard_configs_for_all_models(
        batch_size=1,
        num_batches_inference=150,
        num_batches_training=50,
        dataset_sample_shape=dataset_sample_shape,
        input_sample_shape=INPUT_SAMPLE_SHAPE,
        device_name=device_name,
    )

    return workload_cfgs


def create_standard_configs_for_all_models(
    batch_size: int,
    num_batches_inference: int,
    num_batches_training: int,
    dataset_sample_shape: Sequence[int],
    input_sample_shape: ImageShape,
    device_name: str,
) -> List[AIWorkloadBaseConfig]:

    NUM_CLASSES = 100
    TRAINING_EPOCHS = 5

    workload_cfgs = []
    for model_identifier in ModelIdentifier:

        infer_cfg = InferenceConfig(
            dataset_cfg=DatasetConfig(
                batch_size=batch_size,
                input_shape_without_batch=dataset_sample_shape,
                num_batches=num_batches_inference,
            ),
            model_cfg=ClassificiationModelConfig(
                model_identifier=model_identifier,
                model_shape=input_sample_shape,
                num_classes=NUM_CLASSES,
            ),
            device_name=device_name,
            precision=NumericalPrecision.DEFAULT_PRECISION,
        )
        workload_cfgs.append(infer_cfg)

        train_cfg = TrainingConfig(
            dataset_cfg=DatasetConfig(
                batch_size=batch_size,
                input_shape_without_batch=dataset_sample_shape,
                num_batches=num_batches_training,
            ),
            model_cfg=ClassificiationModelConfig(
                model_identifier=model_identifier,
                model_shape=input_sample_shape,
                num_classes=NUM_CLASSES,
            ),
            device_name=device_name,
            precision=NumericalPrecision.DEFAULT_PRECISION,
            epochs=TRAINING_EPOCHS,
        )
        workload_cfgs.append(train_cfg)

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
