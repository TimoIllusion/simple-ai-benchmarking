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
        if model_cfg.model_identifier == ModelIdentifier.SIMPLE_CLASSIFICATION_CNN:
            from simple_ai_benchmarking.models.pt.simple_classification_cnn import (
                SimpleClassificationCNN,
            )

        # elif model_cfg.model_identifier == ModelIdentifier.VIT_B_16:
        #     import torchvision

        #     resnet50 = torchvision.models.resnet50(num_classes=1000)
        #     vitb16 = AIModelWrapper("ViT-B-16", torchvision.models.vit_b_16(num_classes=1000))

        #     return SimpleClassificationCNN(model_cfg.num_classes, model_cfg.model_shape)
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

        elif model_cfg.model_identifier == ModelIdentifier.VIT_B_16:

            from simple_ai_benchmarking.models.tf.vit import VisionTransformer

            vit_base_16 = VisionTransformer(
                patch_size=16,
                num_patches=196,
                projection_dim=768,
                num_heads=12,
                transformer_units=[3072, 768],
                transformer_layers=12,
                mlp_head_units=[2048, 1024],
                num_classes=model_cfg.num_classes,
            )

            return vit_base_16

        else:
            raise ValueError(
                f"Model {model_cfg.model_identifier} not supported for TensorFlow"
            )
