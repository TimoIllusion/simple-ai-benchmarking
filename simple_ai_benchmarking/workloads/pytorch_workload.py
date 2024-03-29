import platform
import os

from loguru import logger

import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset

from simple_ai_benchmarking.log import *
from simple_ai_benchmarking.workloads.ai_workload import AIWorkload
from simple_ai_benchmarking.definitions import NumericalPrecision


class PyTorchWorkload(AIWorkload):

    def setup(self) -> None:

        print(self.model)
        print("Number of model parameters:", self.count_model_parameters())

        self.device = torch.device(self.cfg.device_name)

        self.inputs, self.targets = self._generate_random_dataset_with_numpy()

        self.inputs = torch.Tensor(self.inputs).to(torch.float32)
        self.targets = torch.Tensor(self.targets).to(torch.int64)

        logger.debug(
            "Synthetic Dataset PyTorch Inputs Shape: {} {}",
            self.inputs.shape,
            self.inputs.dtype,
        )
        logger.debug(
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

    def train(self) -> None:

        if self.cfg.data_type == NumericalPrecision.DEFAULT_PRECISION:
            self._training_loop()
        else:
            with torch.autocast(
                device_type=self.autocast_device_type, dtype=self.numerical_precision
            ):
                self._training_loop()

    def _training_loop(self) -> None:

        self.model.train()

        for epoch in tqdm.tqdm(range(self.cfg.epochs)):
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

    def eval(self) -> None:
        raise NotImplementedError("Eval not implemented yet")

    def infer(self) -> None:

        if self.cfg.data_type == NumericalPrecision.DEFAULT_PRECISION:
            self._infer_loop()
        else:
            with torch.autocast(
                device_type=self.autocast_device_type, dtype=self.numerical_precision
            ):
                self._infer_loop()

    def _infer_loop(self) -> None:

        self.model.eval()

        for inputs, labels in tqdm.tqdm(self.dataloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)

    def _get_accelerator_info(self) -> str:
        if torch.cuda.is_available():
            device_info = str(torch.cuda.get_device_name(None))
        else:
            device_info = ""

        return device_info

    def _get_ai_framework_name(self) -> str:
        return "torch"

    def _get_ai_framework_version(self) -> str:
        return torch.__version__

    def _get_ai_framework_extra_info(self) -> str:
        extra_info = ""
        if "AI_FRAMEWORK_EXTRA_INFO_PT" in os.environ:
            extra_info = os.environ["AI_FRAMEWORK_EXTRA_INFO_PT"]
        return extra_info
