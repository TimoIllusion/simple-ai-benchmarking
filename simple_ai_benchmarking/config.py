from typing import List

from simple_ai_benchmarking.definitions import NumericalPrecision
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
    
    workloads = [
            PyTorchWorkload(
                PTSimpleClassificationCNN.build_model(100, [3,224,224]), 
                10, 
                10, 
                8, 
                device,
                NumericalPrecision.MIXED_FP16
                ),
            PyTorchWorkload(
                PTSimpleClassificationCNN.build_model(100, [3,224,224]), 
                10, 
                10, 
                8, 
                device,
                NumericalPrecision.DEFAULT_PRECISION
                ),
            PyTorchWorkload(
                PTSimpleClassificationCNN.build_model(100, [3,224,224]), 
                10, 
                10, 
                8, 
                device,
                NumericalPrecision.EXPLICIT_FP32
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
    
    # Get more models form keras model zoo: https://keras.io/api/applications/
    workloads = [
        TensorFlowKerasWorkload(
            TFSimpleClassificationCNN.build_model(100, [224,224,3]), 
            10, 
            10, 
            8, 
            device,
            NumericalPrecision.MIXED_FP16
            ), # <1 GB
         TensorFlowKerasWorkload(
            TFSimpleClassificationCNN.build_model(100, [224,224,3]), 
            10, 
            10, 
            8, 
            device,
            NumericalPrecision.DEFAULT_PRECISION,
            ), # <1 GB
        TensorFlowKerasWorkload(
            TFSimpleClassificationCNN.build_model(100, [224,224,3]), 
            10, 
            10, 
            8, 
            device,
            NumericalPrecision.EXPLICIT_FP32,
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