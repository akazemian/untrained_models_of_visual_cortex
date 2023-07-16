from abc import ABC, abstractmethod
import torch    
    

    
            
            
class Regression(ABC):
    @abstractmethod
    def fit(self, *, x: torch.Tensor, y: torch.Tensor) -> None:
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    
    
