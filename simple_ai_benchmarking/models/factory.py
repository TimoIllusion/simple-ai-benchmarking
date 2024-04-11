# Project Name: simple-ai-benchmarking
# File Name: factory.py
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


from simple_ai_benchmarking.config_structures import (
    ModelIdentifier,
    AIFramework,
    ClassificiationModelConfig,
)


class ClassificationModelFactory:

    @staticmethod
    def create_model(
        model_cfg: ClassificiationModelConfig,
        framework: AIFramework,
    ):
        if framework == AIFramework.PYTORCH:
            return ClassificationModelFactory._create_pytorch_model(model_cfg)
        elif framework == AIFramework.TENSORFLOW:
            return ClassificationModelFactory._create_tensorflow_model(model_cfg)
        else:
            raise ValueError(f"Framework {framework} not supported")

    @staticmethod
    def _create_pytorch_model(
        model_cfg: ClassificiationModelConfig,
    ):
        input_shape = model_cfg.model_shape.to_tuple_chw()
        
        if model_cfg.model_identifier == ModelIdentifier.SIMPLE_CLASSIFICATION_CNN:
            from simple_ai_benchmarking.models.pt.simple_classification_cnn import (
                SimpleClassificationCNN,
            )

            return SimpleClassificationCNN(model_cfg.num_classes, input_shape)

        elif model_cfg.model_identifier == ModelIdentifier.VIT_B_16:
            from simple_ai_benchmarking.models.pt.vit import VisionTransformer

            return VisionTransformer(model_cfg.num_classes)

        elif model_cfg.model_identifier == ModelIdentifier.RESNET50:

            from simple_ai_benchmarking.models.pt.resnet50 import ResNet50

            return ResNet50(model_cfg.num_classes)

        else:
            raise ValueError(
                f"Model {model_cfg.model_identifier} not supported for PyTorch"
            )

    @staticmethod
    def _create_tensorflow_model(
        model_cfg: ClassificiationModelConfig,
    ):
        
        input_shape = model_cfg.model_shape.to_tuple_hwc()
        
        if model_cfg.model_identifier == ModelIdentifier.SIMPLE_CLASSIFICATION_CNN:
            from simple_ai_benchmarking.models.tf.simple_classification_cnn import (
                SimpleClassificationCNN,
            )

            return SimpleClassificationCNN(model_cfg.num_classes, input_shape)

        elif model_cfg.model_identifier == ModelIdentifier.VIT_B_16:

            from simple_ai_benchmarking.models.tf.vit import create_vit_b_16

            return create_vit_b_16(model_cfg.num_classes, input_shape)

        elif model_cfg.model_identifier == ModelIdentifier.RESNET50:

            from simple_ai_benchmarking.models.tf.resnet50 import ResNet50
            
            return ResNet50(model_cfg.num_classes, input_shape)

        else:
            raise ValueError(
                f"Model {model_cfg.model_identifier} not supported for TensorFlow"
            )
