import torch
from torch import nn
import numpy as np


class NonLinearity(nn.Module):
    
    def __init__(self,operation):
        super().__init__()
    
        self.operation = operation
        self.operation_type = ['zscore', 'leaky_relu', 'relu', 'gelu', 'abs', 'elu','none']

    
    def forward(self,x):

        assert self.operation in self.operation_type, f'invalid operation type, choose one of {self.operation_type}'
        
        match self.operation:

            case 'zscore':
                std = x.std(dim=1, keepdims=True)
                mean = x.mean(dim=1, keepdims=True)
                x_norm = (x - mean)/std
                return x_norm


            case 'elu':
                nl = nn.ELU(alpha=1.0)
                return nl(x)
        
        
            case 'leaky_relu': 
                nl = nn.LeakyReLU()
                return nl(x)


            case 'relu': 
                nl = nn.ReLU()
                return nl(x)

            
            case 'gelu': 
                nl = nn.GELU()
                return nl(x)

            
            case 'abs': 
                return x.abs()
            
            
            case 'none': 
                return x         