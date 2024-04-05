import platform
import math

from loguru import logger

import torch
from torch.utils.data import DataLoader, TensorDataset

from simple_ai_benchmarking.results import *
from simple_ai_benchmarking.workloads.ai_workload import AIWorkload
from simple_ai_benchmarking.definitions import NumericalPrecision


class PyTorchTraining(AIWorkload):

    def setup(self) -> None:

        # print(self.model)
        logger.trace("Number of model parameters: {}", self.count_model_parameters())

        self.device = torch.device(self.cfg.device_name)

        self.inputs, self.targets = self._generate_random_dataset_with_numpy()

        self.inputs = torch.Tensor(self.inputs).to(torch.float32)
        self.targets = torch.Tensor(self.targets).to(torch.int64)

        logger.trace(
            "Synthetic Dataset PyTorch Inputs Shape: {} {}",
            self.inputs.shape,
            self.inputs.dtype,
        )
        logger.trace(
            "Synthetic Dataset PyTorch Targets Shape: {} {}",
            self.targets.shape,
            self.targets.dtype,
        )

        dataset = TensorDataset(self.inputs, self.targets)
        self.dataloader = DataLoader(
            dataset, batch_size=self.cfg.batch_size, shuffle=False, pin_memory=True
        )

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

        if self.cfg.data_type == NumericalPrecision.MIXED_FP16:
            self.numerical_precision = torch.float16
        elif self.cfg.data_type == NumericalPrecision.DEFAULT_PRECISION:
            pass
        elif self.cfg.data_type == NumericalPrecision.EXPLICIT_FP32:
            self.numerical_precision = torch.float32
        else:
            raise NotImplementedError(
                f"Data type not implemented: {self.cfg.data_type}"
            )

    def _assign_autocast_device_type(self) -> None:
        self.autocast_device_type = "cuda" if "cuda" in self.cfg.device_name else "cpu"

    def count_model_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _warmup(self) -> None:
        self._training_loop(1, max_batches=10)

    def _execute(self) -> None:

        if self.cfg.data_type == NumericalPrecision.DEFAULT_PRECISION:
            self._training_loop(self.cfg.epochs)
        else:
            with torch.autocast(
                device_type=self.autocast_device_type, dtype=self.numerical_precision
            ):
                self._training_loop(self.cfg.epochs)

    def _training_loop(self, max_epochs: int, max_batches: int = math.inf) -> None:

        self.model.train()

        batch_counter = 0

        for _ in range(max_epochs):

            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                batch_counter += 1

                self._increment_iteration_counter_by_batch_size()

                if batch_counter >= max_batches:
                    break

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


class PyTorchInference(PyTorchTraining):

    def _warmup(self) -> None:
        self._infer_loop(max_batches=10)

    def _execute(self) -> None:

        if self.cfg.data_type == NumericalPrecision.DEFAULT_PRECISION:
            self._infer_loop()
        else:
            with torch.autocast(
                device_type=self.autocast_device_type, dtype=self.numerical_precision
            ):
                self._infer_loop()

    def _infer_loop(self, max_batches: int = math.inf) -> None:

        self.model.eval()

        for i, (inputs, labels) in enumerate(self.dataloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)

            self._increment_iteration_counter_by_batch_size()

            if i >= max_batches:
                break
