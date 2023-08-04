from models.layer_operations.convolution import Convolution
from models.layer_operations.output import Output
from models.layer_operations.blurpool import BlurPool
from models.layer_operations.nonlinearity import NonLinearity
import torch
from torch import nn
                         


class Model(nn.Module):
    
    
    def __init__(self,
                lin1: nn.Module,
                 lin2: nn.Module,
                 lin3: nn.Module,
                 nl: nn.Module,
                 last: nn.Module,
                 print_shape:bool=True
                ):
        
        super(Model, self).__init__()
        
        self.lin1 = lin1
        self.lin2 = lin2
        self.lin3 = lin3
        
        self.nl = nl
        self.last = last
        
        self.print_shape = print_shape
        
    def forward(self, x:nn.Module): 
       
        N = x.shape[0]
        x = self.lin1(x.reshape(N,-1))  # linear layer
        x = self.nl(x)
        if self.print_shape:
            print('lin1',x.shape)
        
        x = self.lin2(x)
        x = self.nl(x)
        if self.print_shape:
            print('lin2',x.shape)
            
        x = self.lin3(x)
        x = self.nl(x)
        if self.print_shape:
            print('lin3',x.shape)
            
        x = self.last(x)
    
        return x    


    
  

    
class FCModel3L:


    def __init__(self, 
                 image_size:int = 224,
                 features_1:int = 108,
                 features_2:int = 1000,
                 features_3:int = 10000):    
        
        self.features_1 = features_1
        self.features_2 = features_2
        self.features_3 = features_3
        self.input_dim = image_size**2*3
    
    
    def Build(self):        
        
        lin1 = nn.Linear(self.input_dim, self.features_1)
        lin2 = nn.Linear(self.features_1, self.features_2)
        lin3 = nn.Linear(self.features_2, self.features_3)
        
        
        nl = NonLinearity('abs')
        last = Output()
        
        return Model(lin1, lin2, lin3, nl, last)
    