# Note: This module defines the default configurations for the tensorflow benchmark. By design, this is hardcoded and NOT easily customizable by the user,
# so that the user does not have to worry about the benchmark configuration. However, this may change in the future, since it is not very good programming style.

from typing import List
from copy import copy, deepcopy

import tensorflow as tf

from simple_ai_benchmarking.definitions import (
    NumericalPrecision,
    AIWorkloadBaseConfig,
    AIModelWrapper,
)
from simple_ai_benchmarking.workloads.ai_workload import AIWorkload
from simple_ai_benchmarking.workloads.tensorflow_workload import TensorFlowKerasTraining, TensorFlowKerasInference
from simple_ai_benchmarking.models.tf.simple_classification_cnn import (
    SimpleClassificationCNN,
)


def build_default_tf_workloads() -> List[AIWorkload]:

    device_name = "/gpu:0"

    model_shape = [224, 224, 3]

    common_cfg_default = AIWorkloadBaseConfig(
        batch_size=8,
        num_batches=50,
        epochs=10,
        input_shape_without_batch=model_shape,
        target_shape_without_batch=[],
        device_name=device_name,
        data_type=NumericalPrecision.DEFAULT_PRECISION,
    )

    common_cfg_fp16_mixed = copy(common_cfg_default)
    common_cfg_fp16_mixed.data_type = NumericalPrecision.MIXED_FP16

    # common_cfg_fp32_explicit = copy(common_cfg_default)
    # common_cfg_fp32_explicit.data_type = NumericalPrecision.EXPLICIT_FP32

    common_cfg_bs1_default = copy(common_cfg_default)
    common_cfg_bs1_default.batch_size = 1

    simple_classification_cnn = AIModelWrapper(
        "SimpleClassificationCNN", SimpleClassificationCNN(100, model_shape)
    )
    simple_classification_cnn2 = AIModelWrapper(
        "SimpleClassificationCNN", SimpleClassificationCNN(100, model_shape)
    )
    resnet50 = AIModelWrapper("ResNet50", tf.keras.applications.ResNet50(weights=None))
    # tf.keras.applications.EfficientNetB5()
    # tf.keras.applications.EfficientNetB0(),

    # Get more models form keras model zoo: https://keras.io/api/applications/
    workloads = [
        TensorFlowKerasInference(simple_classification_cnn, common_cfg_default),
        TensorFlowKerasInference(
            simple_classification_cnn2, common_cfg_fp16_mixed
        ),
        TensorFlowKerasInference(resnet50, common_cfg_bs1_default),
        TensorFlowKerasTraining(simple_classification_cnn, common_cfg_default),
        TensorFlowKerasTraining(
            simple_classification_cnn2, common_cfg_fp16_mixed
        ),
        TensorFlowKerasTraining(resnet50, common_cfg_bs1_default),
    ]

    return workloads
