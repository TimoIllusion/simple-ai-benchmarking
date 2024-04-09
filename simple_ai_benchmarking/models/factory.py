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
