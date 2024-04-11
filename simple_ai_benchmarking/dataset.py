# Project Name: simple-ai-benchmarking
# File Name: dataset.py
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


from typing import List, Tuple
from abc import ABC, abstractmethod

from simple_ai_benchmarking.config_structures import DatasetConfig, AIFramework

from loguru import logger

import numpy as np
import psutil


def estimate_array_memory_usage(shape: list, dtype: np.dtype) -> float:
    """
    Estimates the memory usage of a numpy array.

    Parameters:
    - shape: The shape of the array.
    - dtype: The data type of the array.

    Returns:
    - The estimated memory usage in bytes.
    """
    itemsize = np.dtype(dtype).itemsize
    logger.trace(f"Itemsize : {itemsize} Bytes, shape: {shape}")

    num_items = np.prod(shape)

    logger.trace(f"Items: {num_items}")

    return num_items * itemsize


def get_available_memory_in_bytes():
    return psutil.virtual_memory().available


class Dataset(ABC):

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def get_inputs_and_targets(self) -> Tuple[object, object]:
        """Returns dataset.

        Returns:
            Tuple[object, object]: inputs, targets
        """
        pass


class SyntheticDataset(Dataset):

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg

        self.inputs = None
        self.targets = None

    def prepare(self) -> None:

        self.dataset_inputs_shape = [self.cfg.num_batches * self.cfg.batch_size] + list(
            self.cfg.input_shape_without_batch
        )

        self.dataset_targets_shape = [
            self.cfg.num_batches * self.cfg.batch_size
        ] + list(self.cfg.target_shape_without_batch)

        inputs_estimated_size = estimate_array_memory_usage(
            self.dataset_inputs_shape, np.float32
        )
        targets_estimated_size = estimate_array_memory_usage(
            self.dataset_targets_shape, np.int64
        )
        total_estimated_size = inputs_estimated_size + targets_estimated_size

        available_memory = get_available_memory_in_bytes()

        logger.info(f"Estimated memory usage: {total_estimated_size/1e9} GB")
        logger.info(f"Available memory: {available_memory/1e9} GB")

        total_estimated_size_with_overhead = (
            total_estimated_size * 2.0 + 2 * 1e9
        )  # double estimated size and add static overhead

        logger.info(
            f"Estimated memory usage with overhead: {total_estimated_size_with_overhead/1e9} GB"
        )

        if total_estimated_size_with_overhead > available_memory * 0.9:
            raise MemoryError(
                "Insufficient memory, select lower num_batches or batch_size."
            )

        self._create_data()

        logger.debug(
            "Synthetic Dataset Inputs Shape & Type: {} {}",
            self.inputs.shape,
            self.inputs.dtype,
        )
        logger.debug(
            "Synthetic Dataset Targets Shape & Type: {} {}",
            self.targets.shape,
            self.targets.dtype,
        )

    @abstractmethod
    def _create_data(self) -> None:
        pass

    def get_inputs_and_targets(self) -> Tuple[object, object]:
        if self.inputs is None or self.targets is None:
            raise ValueError("Dataset not prepared.")
        return self.inputs, self.targets


class SyntheticPytorchDataset(SyntheticDataset):

    def _create_data(self) -> None:
        import torch

        self.inputs = torch.rand(self.dataset_inputs_shape, dtype=torch.float32).cpu()
        self.targets = torch.randint(
            low=0, high=2, size=self.dataset_targets_shape, dtype=torch.int64
        ).cpu()


class SyntheticTensorFlowDataset(SyntheticDataset):

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg

        self.inputs = None
        self.targets = None

    def _create_data(self):
        import tensorflow as tf

        self.inputs = tf.random.uniform(self.dataset_inputs_shape, dtype=tf.float32)
        self.targets = tf.random.uniform(
            self.dataset_targets_shape, minval=0, maxval=2, dtype=tf.int64
        )


class SyntheticDatasetFactory:

    @staticmethod
    def create_dataset(cfg: DatasetConfig, framework: AIFramework) -> Dataset:
        if framework == AIFramework.PYTORCH:
            return SyntheticPytorchDataset(cfg)
        elif framework == AIFramework.TENSORFLOW:
            return SyntheticTensorFlowDataset(cfg)
        else:
            raise ValueError("Unsupported framework.")
