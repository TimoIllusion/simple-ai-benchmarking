# Project Name: simple-ai-benchmarking
# File Name: config_structures.py
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


from typing import Tuple, Sequence
from enum import Enum
from dataclasses import dataclass, field


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
    GENERATION = "Generation"


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
class ClassificationModelConfig(ModelConfig):
    num_classes: int = 2
    model_shape: ImageShape = field(default_factory=lambda: ImageShape(224, 224, 3))


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
    dataset_cfg: DatasetConfig = field(default_factory=lambda: DatasetConfig())
    model_cfg: ModelConfig = field(default_factory=lambda: ModelConfig())

    def __str__(self):
        return f"{self.model_cfg} with {self.precision.name} on {self.device_name}"


@dataclass
class InferenceConfig(AIWorkloadBaseConfig):
    model_cfg: ClassificationModelConfig = field(
        default_factory=lambda: ClassificationModelConfig()
    )


@dataclass
class TrainingConfig(InferenceConfig):
    epochs: int = 5


@dataclass
class GenerationModelConfig:
    vocab_size: int = 32000
    context_length: int = 4096
    embedding_dim: int = 256
    attention_heads: int = 4
    transformer_layers: int = 4


@dataclass
class LLMGenerationConfig:
    """Config for local generation workloads (sibling to the CV configs).

    Standalone rather than an AIWorkloadBaseConfig subclass: generation has no
    dataset/batch/num_classes notion. Concurrency is the batch size processed in
    a single batched forward pass."""

    device_name: str = "cpu"
    model: str = "SimpleTransformerLM"
    requests: int = 10
    warmup_requests: int = 1
    concurrency: int = 1
    prompt_tokens: int = 128
    generated_tokens: int = 256
    precision: NumericalPrecision = NumericalPrecision.DEFAULT_PRECISION
    compute_precision: str = "FP32"
    quantization: str = "none"
    weight_source: str = "random_weights"
    accelerator: str = "unknown"
    ai_framework_version: str = ""
    ai_framework_extra_info: str = ""
    model_params: int = 0
    model_cfg: GenerationModelConfig = field(
        default_factory=lambda: GenerationModelConfig()
    )

    def __str__(self):
        return (
            f"{self.model} (gen) p{self.prompt_tokens}/g{self.generated_tokens} "
            f"c{self.concurrency} on {self.device_name}"
        )
