from typing import Tuple, Sequence
from enum import Enum
from dataclasses import dataclass


class NumericalPrecision(Enum):
    DEFAULT_PRECISION = 0
    MIXED_FP16 = 1
    EXPLICIT_FP32 = 2


class ModelIdentifier(Enum):
    SIMPLE_CLASSIFICATION_CNN = "SimpleClassificationCNN"
    RESNET50 = "ResNet50"
    VIT_B_16 = "ViT-B-16"


class AIFramework(Enum):
    PYTORCH = "PyTorch"
    TENSORFLOW = "TensorFlow"


class AIStage(Enum):
    INFERENCE = "Inference"
    TRAINING = "Training"


@dataclass
class ImageShape:
    width: int
    height: int
    channels: int

    def to_tuple_hwc(self) -> Tuple[int]:
        return (self.height, self.width, self.channels)

    def to_tuple_chw(self) -> Tuple[int]:
        return (self.channels, self.height, self.width)
    
    def to_tuple_depending_on_framework(self, framework: AIFramework) -> Tuple[int]:
        if framework is AIFramework.PYTORCH:
            return self.to_tuple_chw()
        elif framework is AIFramework.TENSORFLOW:
            return self.to_tuple_hwc()
        else:
            raise ValueError("Invalid framework")


@dataclass
class ModelConfig:
    model_identifier: ModelIdentifier = ModelIdentifier.SIMPLE_CLASSIFICATION_CNN
    model_shape: Sequence[int] = ()
    
    def __str__(self):
        return f"{self.model_identifier.name} {self.model_shape}"


@dataclass
class ClassificiationModelConfig(ModelConfig):
    num_classes: int = 2
    model_shape: ImageShape = ImageShape(224, 224, 3)


@dataclass
class DatasetConfig:
    num_batches: int = 50
    batch_size: int = 1
    input_shape_without_batch: Sequence[int] = ()
    target_shape_without_batch: Sequence[int] = ()


@dataclass
class AIWorkloadBaseConfig:
    device_name: str = "NOT SET"
    precision: NumericalPrecision = NumericalPrecision.DEFAULT_PRECISION
    dataset_cfg: DatasetConfig = DatasetConfig()
    model_cfg: ModelConfig = ModelConfig()
    
    def __str__(self):
        return f"{self.model_cfg} with {self.precision.name} on {self.device_name}"


@dataclass
class InferenceConfig(AIWorkloadBaseConfig):
    dataset_cfg: DatasetConfig = DatasetConfig()
    model_cfg: ClassificiationModelConfig = ClassificiationModelConfig()


@dataclass
class TrainingConfig(InferenceConfig):
    epochs: int = 5
