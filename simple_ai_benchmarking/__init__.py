
from abc import ABC, abstractmethod


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
    def build_log_dict(self):
        pass    
    