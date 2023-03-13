import torch
from torch import nn

class Output(nn.Module):
    
    def __init__(self):
        super().__init__()
                
    def forward(self,x):
        return x#.view(x.shape[0], -1)