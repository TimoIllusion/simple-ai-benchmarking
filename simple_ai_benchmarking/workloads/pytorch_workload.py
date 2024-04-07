import platform
from copy import deepcopy

from loguru import logger

import torch
from torch.utils.data import DataLoader, TensorDataset

from simple_ai_benchmarking.workloads.ai_workload import AIWorkload
from simple_ai_benchmarking.config_structures import (
    NumericalPrecision,
    TrainingConfig,
    DatasetConfig,
    InferenceConfig,
)
from simple_ai_benchmarking.dataset import SyntheticDatasetFactory
from simple_ai_benchmarking.models.factory import ClassificationModelFactory
from simple_ai_benchmarking.config_structures import AIFramework, AIStage


class PyTorchTraining(AIWorkload):

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__(config)

        self.cfg: TrainingConfig  # for type hinting

    def setup(self) -> None:

        self.device = torch.device(self.cfg.device_name)

        self.model = ClassificationModelFactory.create_model(
            self.cfg.model_cfg, AIFramework.PYTORCH
        )
        logger.info("Number of model parameters: {}", self._get_model_parameters())

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.001, momentum=0.9
        )
        self.criterion = torch.nn.CrossEntropyLoss()

        self.model.to(self.device)

        self._compile_model_if_supported()
        self._assign_numerical_precision()
        self._assign_autocast_device_type()

    def _compile_model_if_supported(self) -> None:

        version_str = torch.__version__
        major_version = int(version_str.split(".")[0])

        if major_version >= 2 and platform.system() != "Windows":
            torch.compile(self.model)

    def _assign_numerical_precision(self) -> None:

        if self.cfg.precision == NumericalPrecision.MIXED_FP16:
            self.numerical_precision = torch.float16
        elif self.cfg.precision == NumericalPrecision.DEFAULT_PRECISION:
            pass
        elif self.cfg.precision == NumericalPrecision.EXPLICIT_FP32:
            self.numerical_precision = torch.float32
        else:
            raise NotImplementedError(
                f"Data type not implemented: {self.cfg.precision}"
            )

    # TODO: what happens if device is other than cuda?
    def _assign_autocast_device_type(self) -> None:
        self.autocast_device_type = "cuda" if "cuda" in self.cfg.device_name else "cpu"

    def _get_model_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def _warmup(self) -> None:
        warmup_dataset_cfg = deepcopy(self.cfg.dataset_cfg)
        warmup_dataset_cfg.num_batches = 3

        warmup_dataloader = self._prepare_synthetic_dataset(warmup_dataset_cfg)

        self._training_loop_with_precision_wrapper(warmup_dataloader, 1)

    def _prepare_synthetic_dataset(self, dataset_cfg: DatasetConfig) -> DataLoader:

        dataset = SyntheticDatasetFactory.create_dataset(
            dataset_cfg, AIFramework.PYTORCH
        )
        dataset.prepare()
        inputs, targets = dataset.get_inputs_and_targets()

        logger.trace(
            "Synthetic Dataset PyTorch Inputs Shape: {} {}",
            inputs.shape,
            inputs.dtype,
        )
        logger.trace(
            "Synthetic Dataset PyTorch Targets Shape: {} {}",
            targets.shape,
            targets.dtype,
        )

        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(
            dataset,
            batch_size=dataset_cfg.batch_size,
            shuffle=True,
            pin_memory=True,
        )

        return dataloader

    def prepare_execution(self) -> None:
        self.execution_dataloader = self._prepare_synthetic_dataset(
            self.cfg.dataset_cfg
        )

    def _execute(self) -> None:
        self._training_loop_with_precision_wrapper(
            self.execution_dataloader, self.cfg.epochs
        )

    def _training_loop_with_precision_wrapper(
        self, dataloader: DataLoader, max_epochs: int
    ) -> None:
        if self.cfg.precision == NumericalPrecision.DEFAULT_PRECISION:
            self._training_loop(dataloader, max_epochs)
        else:
            with torch.autocast(
                device_type=self.autocast_device_type, dtype=self.numerical_precision
            ):
                self._training_loop(dataloader, max_epochs)

    def _training_loop(self, dataloader: DataLoader, max_epochs: int) -> None:

        self.model.train()

        for _ in range(max_epochs):

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

    def _calculate_iterations(self) -> int:
        return (
            self.cfg.dataset_cfg.num_batches
            * self.cfg.dataset_cfg.batch_size
            * self.cfg.epochs
        )

    def _get_accelerator_info(self) -> str:
        if torch.cuda.is_available():
            device_info = str(torch.cuda.get_device_name(None))
        else:
            device_info = "CPU"

        return device_info

    def _get_ai_framework_name(self) -> str:
        return "torch"

    def _get_ai_framework_version(self) -> str:
        # remove the second part of version, e.g. 1.8.0+cu111 -> 1.8.0

        version = torch.__version__

        if "cu" in version and "git" not in version:
            return version.split("+")[0]
        else:
            return version

    def _get_ai_framework_extra_info(self) -> str:
        version = torch.__version__

        if "cu" in version and "git" not in version:
            cuda_short_str = version.split("+")[1]
            return cuda_short_str
        else:
            return "N/A"

    def _get_ai_stage(self) -> AIStage:
        return AIStage.TRAINING


class PyTorchInference(PyTorchTraining):

    def __init__(self, config: InferenceConfig) -> None:
        super().__init__(config)

        self.cfg: InferenceConfig  # for type hinting

    def _warmup(self) -> None:
        warmup_dataset_cfg = deepcopy(self.cfg.dataset_cfg)
        warmup_dataset_cfg.num_batches = 3

        warmup_dataloader = self._prepare_synthetic_dataset(warmup_dataset_cfg)

        self._infer_loop_with_precision_wrapper(warmup_dataloader)

    def _execute(self) -> None:
        self._infer_loop_with_precision_wrapper(self.execution_dataloader)

    def _infer_loop_with_precision_wrapper(self, dataloader: DataLoader) -> None:

        if self.cfg.precision == NumericalPrecision.DEFAULT_PRECISION:
            self._infer_loop(dataloader)
        else:
            with torch.autocast(
                device_type=self.autocast_device_type, dtype=self.numerical_precision
            ):
                self._infer_loop(dataloader)

    # TODO: use loop that is similar to real world usage (no dataloader, more like webcam image stream etc.)
    def _infer_loop(self, dataloader: DataLoader) -> None:
        self.model.eval()

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)

    def _calculate_iterations(self) -> int:
        return self.cfg.dataset_cfg.num_batches * self.cfg.dataset_cfg.batch_size

    def _get_ai_stage(self) -> AIStage:
        return AIStage.INFERENCE
