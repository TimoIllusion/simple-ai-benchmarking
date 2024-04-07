import time
from copy import deepcopy

from loguru import logger

import tensorflow as tf
import numpy as np

from simple_ai_benchmarking.config_structures import (
    NumericalPrecision,
    AIStage,
    AIFramework,
    InferenceConfig,
    TrainingConfig,
    DatasetConfig,
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
            logger.info("Number of model parameters: {}", self._get_model_parameters())

    def _warmup(self) -> None:
        dataset_cfg = deepcopy(self.cfg.dataset_cfg)
        dataset_cfg.num_batches = 3

        tf_dataset_warmup = self._prepare_synthetic_dataset(dataset_cfg)
        self._train_loop(tf_dataset_warmup, 1)

    def _prepare_synthetic_dataset(self, dataset_cfg: DatasetConfig) -> tf.data.Dataset:
        with tf.device(
            "/cpu:0"
        ):  # Always generate dataset on system RAM, that is why CPU is forced here

            warmup_dataset = SyntheticDatasetFactory.create_dataset(
                dataset_cfg, AIFramework.TENSORFLOW
            )
            warmup_dataset.prepare()
            inputs, targets = warmup_dataset.get_inputs_and_targets()

            logger.trace(
                "Synthetic Dataset TensorFlow Inputs Shape: {} {}",
                inputs.shape,
                inputs.dtype,
            )
            logger.trace(
                "Synthetic Dataset TensorFlow Targets Shape: {} {}",
                targets.shape,
                targets.dtype,
            )

            tf_dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))

            tf_dataset = tf_dataset.batch(dataset_cfg.batch_size)
            tf_dataset = tf_dataset.shuffle(buffer_size=10000)

            tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

        return tf_dataset

    def prepare_execution(self) -> None:
        self.tf_dataset_execution = self._prepare_synthetic_dataset(
            self.cfg.dataset_cfg
        )

    def _execute(self) -> None:
        self._train_loop(self.tf_dataset_execution, self.cfg.epochs)

    def _train_loop(self, dataset: tf.data.Dataset, epochs: int) -> None:
        with tf.device(self.cfg.device_name):
            self.model.fit(
                dataset,
                epochs=epochs,
                validation_data=None,
                verbose=0,
            )

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

    def _get_model_parameters(self) -> int:
        # Credits to https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model

        trainable_count = np.sum(
            [tf.keras.backend.count_params(p) for p in self.model.trainable_weights]
        )
        non_trainable_count = np.sum(
            [tf.keras.backend.count_params(p) for p in self.model.non_trainable_weights]
        )

        total = trainable_count + non_trainable_count

        return int(total)

    def _calculate_iterations(self) -> int:
        return (
            self.cfg.dataset_cfg.num_batches
            * self.cfg.dataset_cfg.batch_size
            * self.cfg.epochs
        )

    def _get_ai_stage(self) -> AIStage:
        return AIStage.TRAINING

    def clean_up(self) -> None:

        del self.model
        del self.tf_dataset_execution
        tf.keras.backend.clear_session()


class TensorFlowInference(TensorFlowTraining):

    def __init__(self, config: InferenceConfig) -> None:
        super().__init__(config)

        self.cfg: InferenceConfig  # for type hinting

    def _warmup(self) -> None:
        warmup_dataset_cfg = deepcopy(self.cfg.dataset_cfg)
        warmup_dataset_cfg.num_batches = 3

        tf_dataset_warmup = self._prepare_synthetic_dataset(warmup_dataset_cfg)
        self._infer_loop(tf_dataset_warmup)

    def _execute(self) -> None:
        self._infer_loop(self.tf_dataset_execution)

    def _infer_loop(self, dataset: tf.data.Dataset) -> None:
        with tf.device(self.cfg.device_name):
            predictions = self.model.predict(dataset, verbose=0)

    def _calculate_iterations(self) -> int:
        return self.cfg.dataset_cfg.num_batches * self.cfg.dataset_cfg.batch_size

    def _get_ai_stage(self) -> AIStage:
        return AIStage.INFERENCE
