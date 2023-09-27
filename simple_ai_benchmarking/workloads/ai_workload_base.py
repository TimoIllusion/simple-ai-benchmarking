from abc import abstractmethod, ABC

from simple_ai_benchmarking.log import BenchmarkResult

class AIWorkloadBase(ABC):
    
    def __init__(self, model, epochs: int, num_batches: int, batch_size: int, device_name: str):
        self.model = model
        self.epochs = epochs
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.device_name = device_name

    @abstractmethod
    def setup(self):
        pass    
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def eval(self):
        pass    
    
    @abstractmethod
    def predict(self):
        pass
    
    @abstractmethod
    def build_result_log(self) -> BenchmarkResult:
        return BenchmarkResult()    








    