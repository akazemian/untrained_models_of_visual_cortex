from models.engineered_model import Model
from models.layer_operations.convolution import StandardConvolution,RandomProjections
from models.layer_operations.output import Output

from models.layer_operations.convolution import *
from models.layer_operations.output import Output
import torch
from torch import nn
                         


class Model(nn.Module):
    
    
    def __init__(self,
                c3: nn.Module,
                batches_3: int,
                last: nn.Module,
                print_shape: bool = False
                ):
        
        super(Model, self).__init__()
        

        self.c3 = c3
        self.batches_3 = batches_3
        self.last = last
        self.print_shape = print_shape
        
        
    def forward(self, x:nn.Module):
                
        #load second-layer selected channels 
        #Insert code for selecting second-layer selected channels
        #if self.print_shape:
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

    
    def __init__(self, filters_3 = 20000, batches_3=1):
    
        self.filters_3 = filters_3
        self.batches_3 = batches_3
        

    
    def Build(self):

        c3 = StandardConvolution(out_channels=self.filters_3,filter_size=3,filter_type='random',pooling=('max',8))        
        last = Output()

        return Model(c3,self.batches_3,last)  
    
    
    
    

