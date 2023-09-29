from typing import List

from simple_ai_benchmarking.workloads.ai_workload_base import AIWorkloadBase
from simple_ai_benchmarking.log import BenchmarkResult, Logger
from simple_ai_benchmarking.timer import Timer
from simple_ai_benchmarking.definitions import NumericalPrecision


def benchmark(workload: AIWorkloadBase) -> BenchmarkResult:
    
    workload.setup()
     
    with Timer() as t:
        print("TRAINING")
        workload.train()
    training_duration_s = t.duration_s
    
    with Timer() as t:
        print("INFERENCE")
        workload.predict()
    eval_duration_s = t.duration_s

    result_log = workload.build_result_log()
    
    result_log.train_duration_s = training_duration_s
    result_log.eval_duration_s = eval_duration_s
    
    result_log = _add_iterations_per_second(result_log)
    
    return result_log

def _add_iterations_per_second(result: BenchmarkResult) -> BenchmarkResult:
    result.iterations_per_second_inference =  result.num_iterations_eval / result.eval_duration_s
    result.iterations_per_second_training = result.num_iterations_training / result.train_duration_s
    return result

def _proccess_workloads(workloads: List[AIWorkloadBase]) -> List[BenchmarkResult]:
    result_logger = Logger(log_dir="")
    
    for workload in workloads:
        benchmark_result = benchmark(workload)
        result_logger.add_result(benchmark_result)
    
    result_logger.print_info()
    result_logger.save()
        
def run_tf_benchmarks():
    
    import tensorflow as tf
    
    from simple_ai_benchmarking.workloads.tensorflow_workload import TensorFlowKerasWorkload
    from simple_ai_benchmarking.models.tf.simple_classification_cnn import SimpleClassificationCNN
    
    device = "/gpu:0"
    
    # Get more models form keras model zoo: https://keras.io/api/applications/
    workloads = [
        TensorFlowKerasWorkload(
            SimpleClassificationCNN.build_model(100, [224,224,3]), 
            10, 
            10, 
            8, 
            device,
            NumericalPrecision.MIXED_FP16_FP32
            ), # <1 GB
         TensorFlowKerasWorkload(
            SimpleClassificationCNN.build_model(100, [224,224,3]), 
            10, 
            10, 
            8, 
            device,
            NumericalPrecision.DEFAULT_FP32,
            ), # <1 GB
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
    
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    
    workloads = [
            PyTorchSyntheticImageClassification(
                torchvision.models.resnet50(num_classes=10),
                10,
                10,
                8,
                device,
                NumericalPrecision.MIXED_FP16_FP32,
            ),
            PyTorchSyntheticImageClassification(
                torchvision.models.resnet50(num_classes=10),
                10,
                10,
                8,
                device,
                NumericalPrecision.DEFAULT_FP32,
            )
        ]
    
    _proccess_workloads(workloads)
    
    
