from loguru import logger

import tensorflow as tf
import numpy as np

from simple_ai_benchmarking.config_structures import (
    NumericalPrecision,
    AIStage,
    AIFramework,
    InferenceConfig,
    TrainingConfig,
)
from simple_ai_benchmarking.dataset import SyntheticDatasetFactory
from simple_ai_benchmarking.models.factory import ClassificationModelFactory
from simple_ai_benchmarking.workloads.ai_workload import AIWorkload


class TensorFlowTraining(AIWorkload):

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__(config)

        self.cfg: TrainingConfig  # for type hinting

    def setup(self) -> None:

        if self.cfg.precision == NumericalPrecision.MIXED_FP16:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        elif self.cfg.precision == NumericalPrecision.EXPLICIT_FP32:
            tf.keras.mixed_precision.set_global_policy("float32")
        elif self.cfg.precision == NumericalPrecision.DEFAULT_PRECISION:
            pass
        else:
            raise NotImplementedError(
                f"Data type not implemented: {self.cfg.precision}"
            )

        with tf.device(
            "/cpu:0"
        ):  # Always generate dataset on system RAM, that is why CPU is forced here

            dataset = SyntheticDatasetFactory.create_dataset(
                self.cfg.dataset_cfg, AIFramework.TENSORFLOW
            )

            dataset.prepare()
            self.inputs, self.targets = dataset.get_dataset()

            self.tf_dataset = tf.data.Dataset.from_tensor_slices(
                (self.inputs, self.targets)
            )

            self.tf_dataset = self.tf_dataset.shuffle(buffer_size=10000)
            self.tf_dataset = self.tf_dataset.batch(self.cfg.dataset_cfg.batch_size)
            self.tf_dataset = self.tf_dataset.prefetch(tf.data.AUTOTUNE)

        with tf.device(
            self.cfg.device_name
        ):  # Model shall be loaded to accelerator device directly
            self.model = ClassificationModelFactory.create_model(
                self.cfg.model_cfg, AIFramework.TENSORFLOW
            )

            self.model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",  # To use target shape of (N, ) instead of (N, num_classes)
                metrics=["accuracy"],
            )
            # self.model.summary()

    def _warmup(self) -> None:
        with tf.device(self.cfg.device_name):
            self.model.fit(
                self.tf_dataset,
                epochs=1,
                validation_data=None,
                verbose=0,
            )

    def _execute(self) -> None:
        with tf.device(self.cfg.device_name):

            self.model.fit(
                self.tf_dataset,
                epochs=self.cfg.epochs,
                validation_data=None,
                verbose=0,
            )

            for _ in range(self.cfg.epochs * self.cfg.dataset_cfg.num_batches):
                self._increment_iteration_counter_by_batch_size()

    def _get_accelerator_info(self) -> str:

        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) > 0:

            gpu_id = int(self.cfg.device_name.split(":")[1])
            device_infos = tf.config.experimental.get_device_details(gpus[gpu_id])
            details = device_infos["device_name"]
        else:
            details = "CPU"

        return details

    def _get_ai_framework_name(self) -> str:
        return "tensorflow"

    def _get_ai_framework_version(self) -> str:
        return tf.__version__

    def _get_ai_framework_extra_info(self) -> str:
        return "N/A"

    @staticmethod
    def get_model_memory_usage(batch_size, model) -> float:
        # Credits to https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model

        shapes_mem_count = 0
        internal_model_mem_count = 0
        for l in model.layers:
            layer_type = l.__class__.__name__
            if layer_type == "Model":
                internal_model_mem_count += TensorFlowTraining.get_model_memory_usage(
                    batch_size, l
                )
            single_layer_mem = 1
            out_shape = l.output_shape
            if type(out_shape) is list:
                out_shape = out_shape[0]
            for s in out_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem

        trainable_count = np.sum(
            [tf.keras.backend.count_params(p) for p in model.trainable_weights]
        )
        non_trainable_count = np.sum(
            [tf.keras.backend.count_params(p) for p in model.non_trainable_weights]
        )

        number_size = 4.0
        if tf.keras.backend.floatx() == "float16":
            number_size = 2.0
        if tf.keras.backend.floatx() == "float64":
            number_size = 8.0

        total_memory = number_size * (
            batch_size * shapes_mem_count + trainable_count + non_trainable_count
        )
        gbytes = np.round(total_memory / (1024.0**3), 3) + internal_model_mem_count
        return gbytes

    def _get_ai_stage(self) -> AIStage:
        return AIStage.TRAINING


class TensorFlowInference(TensorFlowTraining):

    def __init__(self, config: InferenceConfig) -> None:
        super().__init__(config)

        self.cfg: InferenceConfig  # for type hinting

    def _warmup(self) -> None:
        self._infer_loop()

    def _execute(self) -> None:
        self._infer_loop()

    def _infer_loop(self) -> None:
        with tf.device(self.cfg.device_name):

            predictions = self.model.predict(self.tf_dataset, verbose=0)

            for _ in range(self.cfg.dataset_cfg.num_batches):
                self._increment_iteration_counter_by_batch_size()

    def _get_ai_stage(self) -> AIStage:
        return AIStage.INFERENCE
