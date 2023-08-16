from model_features.layer_operations.convolution import Convolution
from model_features.layer_operations.output import Output
from model_features.layer_operations.blurpool import BlurPool
from model_features.layer_operations.nonlinearity import NonLinearity
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


    
    
  

    
class FullyConnected:


    def __init__(self, 
                 image_size:int = 224,
                 features:int = 10000):    
        
        self.features = features
        self.input_dim = image_size**2*3
    
    
    def Build(self):        
        
        lin = nn.Linear(self.input_dim, self.features)
        nl = NonLinearity('relu')
        last = Output()
        
        return Model(lin, nl, last)
    