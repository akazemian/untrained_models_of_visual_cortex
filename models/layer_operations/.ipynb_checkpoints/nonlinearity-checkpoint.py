import torch
from torch import nn
import numpy as np


class nonlinearity(nn.Module):
    
    def __init__(self,operation):
        super().__init__()
    
        self.operation = operation

    
    def forward(self,x):

        if self.operation == 'zscore':
            x = x.data.cpu().numpy()
            std = (np.std(x, axis=1, keepdims=True))
            mean = np.mean(x, axis=1, keepdims=True)
            x_norm = (x - mean)/std
            return torch.Tensor(x_norm)

        if self.operation == 'norm':
            x = x.data.cpu().numpy()
            std = 1
            mean = 0
            x_norm = (x - mean)/std
            return torch.Tensor(x_norm)
        
        if self.operation == 'relu': 
            nl = nn.ReLU()
            return nl(x)

        if self.operation == 'gelu': 
            nl = nn.GELU()
            return nl(x)

        if self.operation == 'abs': 
            return x.abs()