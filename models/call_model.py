from models.engineered_model import Model
from models.layer_operations.convolution import StandardConvolution,RandomProjections
from models.layer_operations.output import Output

from models.layer_operations.convolution import *
from models.layer_operations.output import Output
import torch
from torch import nn


class EngineeredModel:
    
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
                 filters_2=2000,filters_3=20000):
    
        
        self.curv_params = curv_params
        self.filters_1 = self.curv_params['n_ories']*self.curv_params['n_curves']*len(self.curv_params['gau_sizes']*len(self.curv_params['spatial_fre']))
        self.filters_2 = filters_2
        self.filters_3 = filters_3
        
    
    
    
    def Build(self):
    
        c1 = StandardConvolution(filter_size=45,filter_type='curvature',curv_params=self.curv_params)     
        mp1 = nn.MaxPool2d(kernel_size=6)
        c2 = nn.Conv2d(24, self.filters_2, kernel_size=(9, 9), stride=(1, 1), device='cuda')
        mp2 = nn.MaxPool2d(kernel_size=2)
        c3 = nn.Conv2d(self.filters_2, self.filters_3, kernel_size=(3, 3), stride=(1, 1), device='cuda')

        last = Output()

        return Model(c1,mp1,c2,mp2,c3,last)  
    