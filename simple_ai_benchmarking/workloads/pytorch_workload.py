import platform
import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset

from simple_ai_benchmarking.log import BenchmarkResult
from simple_ai_benchmarking.workloads.ai_workload_base import AIWorkloadBase
from simple_ai_benchmarking.definitions import NumericalPrecision

class PyTorchSyntheticImageClassification(AIWorkloadBase):

    def setup(self):
        self.device = torch.device(self.device_name)
        
        synthetic_data = torch.randn(self.num_batches * self.batch_size, 3, 224, 224, dtype=torch.float32)
        synthetic_labels = torch.randint(0, 1, (self.num_batches * self.batch_size,))

        dataset = TensorDataset(synthetic_data, synthetic_labels)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.model.to(self.device)

        self._compile_model_if_supported()
        self._assign_numerical_precision()
        self._assign_autocast_device_type()
        
        
    def _compile_model_if_supported(self):
        version_str = torch.__version__
        major_version = int(version_str.split('.')[0])

        if major_version >= 2 and platform.system() != 'Windows':
            torch.compile(self.model) 

    def _assign_numerical_precision(self):
        if self.data_type == NumericalPrecision.MIXED_FP16_FP32:
            self.numerical_precision = torch.float16
        elif self.data_type == NumericalPrecision.DEFAULT_FP32:
            self.numerical_precision = torch.float32
        else:
            raise NotImplementedError(f"Data type not implemented: {self.data_type}")

    def _assign_autocast_device_type(self):
        self.autocast_device_type =  "cuda" if "cuda" in self.device_name else "cpu"

    def train(self):
        self.model.train()
        running_loss = 0.0

        with torch.autocast(device_type=self.autocast_device_type, dtype=self.numerical_precision):

            for epoch in tqdm.tqdm(range(self.epochs)):
                for inputs, labels in self.dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()

    def eval(self):
        raise NotImplementedError("Eval not implemented yet")

    def predict(self):
        self.model.eval()

        with torch.autocast(device_type=self.autocast_device_type, dtype=self.numerical_precision):
        
            for inputs, labels in tqdm.tqdm(self.dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

    def build_result_log(self) -> BenchmarkResult:
        
        if torch.cuda.is_available():
            device_info = str(torch.cuda.get_device_name(0))
        else:
            device_info = "CPU:" + platform.processor()
        
        benchmark_result = BenchmarkResult(
            self.__class__.__name__,
            "torch-" + torch.__version__,
            device_info,
            self.data_type.name,
            self.batch_size,
            len(self.dataloader.dataset) * self.epochs,
            self.batch_size,
            len(self.dataloader.dataset),
            None,
            None,
            None,
            None,
            None
            )
        
        return benchmark_result
