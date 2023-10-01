from typing import List
from copy import copy

from simple_ai_benchmarking.definitions import *
from simple_ai_benchmarking.workloads.ai_workload_base import AIWorkloadBase

def build_default_pt_workloads() -> List[AIWorkloadBase]:
    
    import torch
    import torchvision

    from simple_ai_benchmarking.workloads.pytorch_workload import PyTorchWorkload
    from simple_ai_benchmarking.models.pt.simple_classification_cnn import PTSimpleClassificationCNN
    
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    
    model_shape = [3,224,224]
    
    common_cfg_default = AIWorkloadBaseConfig(
        batch_size=8,
        num_batches=10,
        epochs=10,
        input_shape_without_batch=model_shape,
        target_shape_without_batch=[],
        device_name=device,
        data_type=NumericalPrecision.DEFAULT_PRECISION,
        )
    
    common_cfg_fp16_mixed = copy(common_cfg_default)
    common_cfg_fp16_mixed.data_type = NumericalPrecision.MIXED_FP16
    
    common_cfg_fp32_explicit = copy(common_cfg_default)
    common_cfg_fp32_explicit.data_type = NumericalPrecision.EXPLICIT_FP32
    
    workloads = [
            PyTorchWorkload(
                PTSimpleClassificationCNN(100, model_shape), 
                common_cfg_default
                ),
            PyTorchWorkload(
                PTSimpleClassificationCNN(100, model_shape), 
                common_cfg_fp16_mixed
                ),
            PyTorchWorkload(
                PTSimpleClassificationCNN(100, model_shape), 
                common_cfg_fp32_explicit
                ),
            # PyTorchSyntheticImageClassification(
            #     torchvision.models.resnet50(num_classes=1000),
            #     10,
            #     10,
            #     8,
            #     device,
            #     NumericalPrecision.MIXED_FP16_FP32,
            # ),
            # PyTorchSyntheticImageClassification(
            #     torchvision.models.resnet50(num_classes=1000),
            #     10,
            #     10,
            #     8,
            #     device,
            #     NumericalPrecision.DEFAULT_FP32,
            # )
        ]
    
    return workloads
    
def build_default_tf_workloads() -> List[AIWorkloadBase]:
    
    import tensorflow as tf
    
    from simple_ai_benchmarking.workloads.tensorflow_workload import TensorFlowKerasWorkload
    from simple_ai_benchmarking.models.tf.simple_classification_cnn import TFSimpleClassificationCNN
    
    device = "/gpu:0"
    
    model_shape = [224,224,3]
    
    common_cfg_default = AIWorkloadBaseConfig(
        batch_size=8,
        num_batches=10,
        epochs=10,
        input_shape_without_batch=model_shape,
        target_shape_without_batch=[],
        device_name=device,
        data_type=NumericalPrecision.DEFAULT_PRECISION,
        )
    
    common_cfg_fp16_mixed = copy(common_cfg_default)
    common_cfg_fp16_mixed.data_type = NumericalPrecision.MIXED_FP16
    
    common_cfg_fp32_explicit = copy(common_cfg_default)
    common_cfg_fp32_explicit.data_type = NumericalPrecision.EXPLICIT_FP32
    
    # Get more models form keras model zoo: https://keras.io/api/applications/
    workloads = [
        TensorFlowKerasWorkload(
            TFSimpleClassificationCNN(100, model_shape), 
            common_cfg_default
            ), # <1 GB
         TensorFlowKerasWorkload(
            TFSimpleClassificationCNN(100, model_shape), 
            common_cfg_fp16_mixed
            ), # <1 GB
        TensorFlowKerasWorkload(
            TFSimpleClassificationCNN(100, model_shape), 
            common_cfg_fp32_explicit
            ), # <1 GB
        # TensorFlowKerasWorkload(
        #     tf.keras.applications.ResNet50(weights=None),
        #     10, 
        #     10, 
        #     8, 
        #     device,
        #     NumericalPrecision.MIXED_FP16_FP32,
        #     ),
        # TensorFlowKerasWorkload(
        #     tf.keras.applications.ResNet50(weights=None),
        #     10, 
        #     10, 
        #     8, 
        #     device,
        #     NumericalPrecision.DEFAULT_FP32,
        #     ),
        # TensorFlowKerasWorkload(
        #     tf.keras.applications.EfficientNetB5(),
        #     10, 
        #     10, 
        #     8, 
        #     device,
        #     ), # ~11 GB
        # TensorFlowKerasWorkload(
        #     tf.keras.applications.EfficientNetB0(), 
        #     10, 
        #     10, 
        #     8, 
        #     device,
        #     ), # ~1 GB
        ]
    
    return workloads