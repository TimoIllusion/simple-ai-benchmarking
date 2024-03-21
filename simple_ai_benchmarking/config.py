# Note: This module defines the default configurations for the benchmark for tensorflow and pytorch. By design, this is hardcoded and NOT easily customizable by the user,
# so that the user does not have to worry about the benchmark configuration. However, this may change in the future, since it is not very good programming style.

from typing import List
from copy import copy
from sys import platform

from simple_ai_benchmarking.definitions import NumericalPrecision, AIWorkloadBaseConfig
from simple_ai_benchmarking.workloads.ai_workload import AIWorkload



  

# TODO: use workload factory to create workloads
def build_default_pt_workloads() -> List[AIWorkload]:
    
    import torch
    import torchvision

    from simple_ai_benchmarking.workloads.pytorch_workload import PyTorchWorkload
    from simple_ai_benchmarking.models.pt.simple_classification_cnn import PTSimpleClassificationCNN
        
    if platform == "linux" or platform == "linux2" or platform == "win32":
            
        if torch.cuda.is_available():
            device_name = "cuda"
        else:
            device_name = "cpu"
            
    elif platform == "darwin":
        
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device_name = "mps"
        else:
            device_name = "cpu"

    else:
        device_name = "cpu"
        
        return device_name
    
    model_shape = [3,224,224]
    
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
                torchvision.models.resnet50(num_classes=1000),
                common_cfg_bs1_default
                ),
        ]
    
    return workloads
    
def build_default_tf_workloads() -> List[AIWorkload]:
    
    import tensorflow as tf
    
    from simple_ai_benchmarking.workloads.tensorflow_workload import TensorFlowKerasWorkload
    from simple_ai_benchmarking.models.tf.simple_classification_cnn import TFSimpleClassificationCNN
    
    device_name = "/gpu:0"
    
    model_shape = [224,224,3]
    
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
        # TensorFlowKerasWorkload(
        #     TFSimpleClassificationCNN(100, model_shape), 
        #     common_cfg_fp32_explicit
        #     ), # <1 GB
        TensorFlowKerasWorkload(
            tf.keras.applications.ResNet50(weights=None),
            common_cfg_bs1_default
            ),
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