from enum import Enum
from dataclasses import dataclass


class NumericalPrecision(Enum):
    DEFAULT_PRECISION = 0
    MIXED_FP16 = 1
    EXPLICIT_FP32 = 2


@dataclass
class AIWorkloadBaseConfig:
    epochs: int
    num_batches: int
    batch_size: int
    device_name: str
    data_type: NumericalPrecision
    input_shape_without_batch: list
    target_shape_without_batch: list
