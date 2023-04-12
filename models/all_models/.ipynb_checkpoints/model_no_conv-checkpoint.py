from models.engineered_model import Model
from models.layer_operations.convolution import StandardConvolution,RandomProjections
from models.layer_operations.output import Output

from models.layer_operations.convolution import *
from models.layer_operations.output import Output
import torch
from torch import nn
                         


class Model(nn.Module):
    
    
    def __init__(self,
                fc1: nn.Module,
                fc2: nn.Module,
                fc3: nn.Module,
                last: nn.Module,
                print_shape: bool = False
                ):
        
        super(Model, self).__init__()
        

        self.fc1 = fc1
        self.fc2 = fc2
        self.fc3 = fc3
        self.last = last
        self.print_shape = print_shape
        
        
    def forward(self, x:nn.Module):
                

        x = x.float()
            
        #conv layer 2
        x = self.c3(x)
        if self.print_shape:
            print('conv3', x.shape)
            
            
        x = self.last(x)
        if self.print_shape:
            print('output', x.shape)
        
        return x    



  


class EngineeredModel3L:

    
    def __init__(self):
        

    
    def Build(self):

        fc1 = nn.Linear(9216, 128)   
        fc2 = nn.Linear(9216, 128)  
        fc3 = nn.Linear(9216, 128)  
        
        last = Output()

        return Model(fc1,fc2,fc3,last)  
    
    
    
    

