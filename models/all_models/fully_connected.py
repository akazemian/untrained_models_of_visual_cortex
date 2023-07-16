from models.layer_operations.convolution import Convolution
from models.layer_operations.output import Output
from models.layer_operations.blurpool import BlurPool
from models.layer_operations.nonlinearity import NonLinearity
import torch
from torch import nn
                         


class Model(nn.Module):
    
    
    def __init__(self,
                lin: nn.Module,
                 nl: nn.Module,
                 last: nn.Module
                ):
        
        super(Model, self).__init__()
        
        self.lin = lin
        self.nl = nl
        self.last = last
        
        
    def forward(self, x:nn.Module): 
       
        N = x.shape[0]
        x = self.lin(x.reshape(N,-1))  # linear layer
        x = self.nl(x)
        x = self.last(x)
    
        return x    


    
    
  

    
class FCModel:


    def __init__(self, 
                 image_size:int = 224,
                 num_features:int = 10000):    
        
        self.num_features = num_features
        self.input_dim = image_size**2
    
    
    def Build(self):        
        
        lin = nn.Linear(self.input_dim, self.num_features)
        nl = NonLinearity('abs')
        last = Output()
        
        return Model(lin, nl, last)
    