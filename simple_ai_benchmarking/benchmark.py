from typing import List

from simple_ai_benchmarking.workloads.ai_workload_base import AIWorkloadBase
from simple_ai_benchmarking.log import *
from simple_ai_benchmarking.timer import Timer
from simple_ai_benchmarking.definitions import NumericalPrecision


def benchmark(workload: AIWorkloadBase) -> BenchmarkResult:
    
    workload.setup()
    
    print("WARMUP")
    workload.warmup()
     
    with Timer() as t:
        print("TRAINING")
        workload.train()
    training_duration_s = t.duration_s
    
    with Timer() as t:
        print("INFERENCE")
        workload.infer()
    infer_duration_s = t.duration_s

    result_log = workload.build_result_log()
    
    result_log.update_train_performance_duration(training_duration_s)
    result_log.update_infer_performance_duration(infer_duration_s)
    
    return result_log

def _proccess_workloads(workloads: List[AIWorkloadBase]) -> List[BenchmarkResult]:
    result_logger = BenchmarkLogger()
    
    for workload in workloads:
        benchmark_result = benchmark(workload)
        result_logger.add_result(benchmark_result)
    
    result_logger.pretty_print_summary()
    
    result_logger.export_to_csv("benchmark_results.csv")
    try:
        result_logger.export_to_excel("benchmark_results.xlsx")
    except ModuleNotFoundError as e:
        print("Could not export to excel:", e, "\nPlease install openpyxl to export to excel, e.g. via SAI [xlsx] extra.")
    
def run_tf_benchmarks():
    
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
    
    _proccess_workloads(workloads)
    
def run_pt_benchmarks():
    
    import torch
    import torchvision

    from simple_ai_benchmarking.workloads.pytorch_workload import PyTorchSyntheticImageClassification
    from simple_ai_benchmarking.models.pt.simple_classification_cnn import PTSimpleClassificationCNN
    
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    
    workloads = [
            PyTorchSyntheticImageClassification(
                PTSimpleClassificationCNN.build_model(100, [3,224,224]), 
                10, 
                10, 
                8, 
                device,
                NumericalPrecision.MIXED_FP16
                ),
            PyTorchSyntheticImageClassification(
                PTSimpleClassificationCNN.build_model(100, [3,224,224]), 
                10, 
                10, 
                8, 
                device,
                NumericalPrecision.DEFAULT_PRECISION
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
    
    _proccess_workloads(workloads)
    
    
