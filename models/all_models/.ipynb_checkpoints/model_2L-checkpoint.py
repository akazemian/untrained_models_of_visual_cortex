from models.engineered_model import Model
from models.layer_operations.convolution import StandardConvolution,RandomProjections
from models.layer_operations.output import Output

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
                print_shape: bool = False
                ):
        
        super(Model, self).__init__()
        
        
        self.c1 = c1 
        self.c2 = c2
        self.batches_2 = batches_2
        self.last = last
        self.mp = nn.MaxPool2d(2)
        self.print_shape = print_shape
        
        
    def forward(self, x:nn.Module):
                
        
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



  
class EngineeredModel2L:
    
    """
    Used to Initialize the Engineered Model
    
    Attributes
    ----------
    curv_params : dict
        the parameters used for creating the gabor filters. The number of filters in this layer = n_ories x n_curves x number of frequencies
    
    filters_2 : str
        number of random filters used in conv layer 2
    
    batches_2 : str 
        the number of batches used to apply conv layer 2 filters. Can be used for larger number of filters to avoid memory issues 
    """
    
    def __init__(self, curv_params = {'n_ories':8,'n_curves':3,'gau_sizes':(5,),'spatial_fre':[1.2]},
                 filters_2=20000,batches_2=1):
    
        
        self.curv_params = curv_params
        self.filters_1 = self.curv_params['n_ories']*self.curv_params['n_curves']*len(self.curv_params['gau_sizes']*len(self.curv_params['spatial_fre']))
        self.filters_2 = filters_2
        self.batches_2 = batches_2
        
    
    
    
    def Build(self):
    
        c1 = StandardConvolution(filter_size=45,filter_type='curvature',pooling=('max',6),curv_params=self.curv_params)     
        c2 = StandardConvolution(out_channels=self.filters_2,filter_size=9,filter_type='random',pooling=('max',2))
        last = Output()

        return Model(c1,c2,self.batches_2,last)  
    