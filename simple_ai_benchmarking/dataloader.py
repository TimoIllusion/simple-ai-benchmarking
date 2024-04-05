from typing import List, Tuple
from abc import ABC

from loguru import logger

import numpy as np


@dataclass
class DataLoaderConfig:
    num_batches: int
    batch_size: int
    input_shape_without_batch: Tuple[int]
    target_shape_without_batch: Tuple[int]

class DataLoader(ABC):
    
    def __init__(self, cfg: DataLoaderConfig):
        self.cfg = cfg

    @abstractmethod
    def prefetch(self):
        pass
    
    @abstractmethod
    def get_batch(self, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        pass

class SyntheticDataLoader(DataLoader):

    def _generate_random_dataset_with_numpy(self) -> Tuple[np.ndarray, np.ndarray]:

        self.dataset_inputs_shape = [self.cfg.num_batches * self.cfg.batch_size] + list(
            self.cfg.input_shape_without_batch
        )
        self.dataset_targets_shape = [
            self.cfg.num_batches * self.cfg.batch_size
        ] + list(self.cfg.target_shape_without_batch)

        inputs = np.random.random(self.dataset_inputs_shape).astype(np.float32)
        targets = np.random.randint(
            low=0, high=2, size=self.dataset_targets_shape
        ).astype(np.int64)

        logger.debug(
            "Synthetic Dataset NumPy Inputs Shape: {} {}", inputs.shape, inputs.dtype
        )
        logger.debug(
            "Synthetic Dataset NumPy Targets Shape: {} {}", targets.shape, targets.dtype
        )

        return inputs, targets