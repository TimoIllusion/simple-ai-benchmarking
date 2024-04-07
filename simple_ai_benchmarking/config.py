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


DEFAULT_INPUT_IMAGE_SHAPE = ImageShape(224, 224, 3)
DEFAULT_NUM_CLASSES = 100
DEFAULT_EPOCHS = 5


@dataclass
class ModelConfig:
    model_identifier: ModelIdentifier = "NOT SET"
    model_shape: Sequence[int] = ()


@dataclass
class ClassificiationModelConfig(ModelConfig):
    num_classes: int = 2


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


@dataclass
class PytorchInferenceConfig(AIWorkloadBaseConfig):
    dataset_cfg: DatasetConfig = DatasetConfig(
        input_shape_without_batch=DEFAULT_INPUT_IMAGE_SHAPE.to_tuple_chw()
    )
    model_cfg: ClassificiationModelConfig = ClassificiationModelConfig(
        model_identifier=ModelIdentifier.SIMPLE_CLASSIFICATION_CNN,
        model_shape=DEFAULT_INPUT_IMAGE_SHAPE.to_tuple_chw(),
        num_classes=DEFAULT_NUM_CLASSES,
    )


@dataclass
class PytorchTrainingConfig(PytorchInferenceConfig):
    epochs: int = DEFAULT_EPOCHS


@dataclass
class TensorflowInferenceConfig(AIWorkloadBaseConfig):
    dataset_cfg: DatasetConfig = DatasetConfig(
        input_shape_without_batch=DEFAULT_INPUT_IMAGE_SHAPE.to_tuple_hwc()
    )
    model_cfg: ClassificiationModelConfig = ClassificiationModelConfig(
        model_identifier=ModelIdentifier.SIMPLE_CLASSIFICATION_CNN,
        model_shape=DEFAULT_INPUT_IMAGE_SHAPE.to_tuple_hwc(),
        num_classes=DEFAULT_NUM_CLASSES,
    )


@dataclass
class TensorflowTrainingConfig(TensorflowInferenceConfig):
    epochs: int = DEFAULT_EPOCHS
