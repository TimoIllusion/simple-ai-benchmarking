from typing import List, Tuple
from abc import ABC, abstractmethod

from simple_ai_benchmarking.config_structures import DatasetConfig, AIFramework

from loguru import logger

import numpy as np
import psutil


def _estimate_array_memory_usage(shape: list, dtype: np.dtype) -> float:
    """
    Estimates the memory usage of a numpy array.

    Parameters:
    - shape: The shape of the array.
    - dtype: The data type of the array.

    Returns:
    - The estimated memory usage in bytes.
    """
    itemsize = np.dtype(dtype).itemsize
    return np.prod(shape) * itemsize


def _is_memory_sufficient_for_arrays(*arrays_info) -> bool:
    """
    Checks if there is enough available system memory to create the numpy arrays.

    Parameters:
    - arrays_info: A list of tuples where each tuple contains the shape and dtype of the intended array.

    Returns:
    - True if there is enough memory to create the arrays, False otherwise.
    """
    required_memory = sum(
        _estimate_array_memory_usage(shape, dtype) for shape, dtype in arrays_info
    )
    available_memory = psutil.virtual_memory().available

    logger.warning(f"{required_memory/1e9} {available_memory/1e9}")
    logger.info(required_memory < available_memory)

    return required_memory < available_memory


class Dataset(ABC):

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def get_dataset(self) -> Tuple[object, object]:
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

        if _is_memory_sufficient_for_arrays(
            (self.dataset_inputs_shape, np.float32),
            (self.dataset_targets_shape, np.int64),
        ):

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
        else:
            raise MemoryError(
                "Insufficient memory, select lower num_batches or batch_size."
            )

    @abstractmethod
    def _create_data(self) -> None:
        pass

    def get_dataset(self) -> Tuple[object, object]:
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
