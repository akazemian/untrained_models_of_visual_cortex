from models.layer_operations.convolution import *
from models.layer_operations.output import Output
import torch
from torch import nn
                         


class Model(nn.Module):
    
    
    def __init__(self,
                c1: nn.Module,
                c2: nn.Module,
                batches_2: int,
                last: nn.Module,
                print_shape: bool = True
                ):
        
        super(Model, self).__init__()
        
        
        self.c1 = c1 
        self.c2 = c2
        self.batches_2 = batches_2
        self.last = last
        self.mp = nn.MaxPool2d(2)
        self.print_shape = print_shape
        
        
    def forward(self, x:nn.Module):
                
        print('image',x.shape)
        
        #conv layer 1
        x = self.c1(x)
        if self.print_shape:
            print('conv1', x.shape)
    
        
        #conv layer 2
        conv_2 = []
        for i in range(self.batches_2):
            conv_2.append(self.c2(x)) 
        x = torch.cat(conv_2,dim=1)
        if self.print_shape:
            print('conv2', x.shape)
            
        x = self.last(x)
        if self.print_shape:
            print('output', x.shape)
        
        return x    



  