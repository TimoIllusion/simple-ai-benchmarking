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
    logger.warning(f"Itemsize : {itemsize} Bytes, shape: {shape}")
    
    num_items = np.prod(shape)
    
    logger.warning(f"Items: {num_items}")
                   
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

        self.dataset_targets_shape = [self.cfg.num_batches * self.cfg.batch_size] + list(
            self.cfg.target_shape_without_batch
        )

        inputs_estimated_size = estimate_array_memory_usage(
            self.dataset_inputs_shape, np.float32
        )
        targets_estimated_size = estimate_array_memory_usage(
            self.dataset_targets_shape, np.int64
        )
        total_estimated_size = inputs_estimated_size + targets_estimated_size

        available_memory = get_available_memory_in_bytes()

        logger.warning(f"Estimated memory usage: {total_estimated_size/1e9} GB")
        logger.warning(f"Available memory: {available_memory/1e9} GB")

        if total_estimated_size > available_memory * 0.9:
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
