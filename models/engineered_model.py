from models.layer_operations.convolution import StandardConvolution,RandomProjections
from models.layer_operations.output import Output

from models.layer_operations.convolution import *
from models.layer_operations.output import Output
import torch
from torch import nn
                         


class Model(nn.Module):
    
    
    def __init__(self,
                c1: nn.Module,
                mp1: nn.Module,
                c2: nn.Module,
                mp2: nn.Module,
                c3: nn.Module,
                last: nn.Module,
                print_shape: bool = False
                ):
        
        super(Model, self).__init__()
        
        
        self.c1 = c1 
        self.mp1 = mp1
        self.c2 = c2
        self.mp2 = mp2
        self.c3 = c3
        self.last = last
        self.print_shape = print_shape
        
        
    def forward(self, x:nn.Module):
                
        
        #conv layer 1
        x = self.c1(x)
        if self.print_shape:
            print('conv1', x.shape)
    
        x = self.mp1(x)
        
        #conv layer 2
        x = self.c2(x)
        if self.print_shape:
            print('conv2', x.shape)
            
        x = self.mp2(x)
        
        #conv layer 3
        x = self.c3(x)
        if self.print_shape:
            print('conv3', x.shape)
            
        
        x = self.last(x)
        if self.print_shape:
            print('output', x.shape)
        
        return x 