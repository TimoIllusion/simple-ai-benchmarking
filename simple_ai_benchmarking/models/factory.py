from simple_ai_benchmarking.config import (
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
        if model_cfg.model_identifier == ModelIdentifier.SIMPLE_CLASSIFICATION_CNN:
            from simple_ai_benchmarking.models.pt.simple_classification_cnn import (
                SimpleClassificationCNN,
            )

            return SimpleClassificationCNN(model_cfg.num_classes, model_cfg.model_shape)
        else:
            raise ValueError(
                f"Model {model_cfg.model_identifier} not supported for PyTorch"
            )

    @staticmethod
    def _create_tensorflow_model(
        model_cfg: ClassificiationModelConfig,
    ):
        if model_cfg.model_identifier == ModelIdentifier.SIMPLE_CLASSIFICATION_CNN:
            from simple_ai_benchmarking.models.tf.simple_classification_cnn import (
                SimpleClassificationCNN,
            )

            return SimpleClassificationCNN(model_cfg.num_classes, model_cfg.model_shape)
        else:
            raise ValueError(
                f"Model {model_cfg.model_identifier} not supported for TensorFlow"
            )
