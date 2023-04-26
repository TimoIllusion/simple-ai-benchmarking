
from abc import ABC, abstractmethod
from simple_ai_benchmarking.log import BenchmarkResult

#TODO: add TensorFlowAIWorkload as subclass
class AIWorkload(ABC):
    
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
    def build_result_log(self) -> BenchmarkResult:
        return BenchmarkResult()    
    