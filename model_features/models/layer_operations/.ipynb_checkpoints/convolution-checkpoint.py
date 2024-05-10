from .preset_filters import filters
from torch.nn import functional as F
import math
import torch
torch.manual_seed(42)
from torch import nn
import numpy as np
import pywt


class WaveletConvolution(nn.Module):
    
    
    def __init__(self, 
                 filter_type:str,
                 filter_params:dict=None,
                 filter_size:int=None,
                 device:str=None

                ):
                
        super().__init__()
        

        self.filter_type = filter_type
        self.filter_size = 15
        self.filter_params = get_kernel_params(self.filter_type)
        self.layer_size = get_layer_size(self.filter_type, self.filter_params)
        self.device = device
        
    
    def forward(self,x):
            
        x = x.to(self.device)
        
        in_channels = x.shape[1]
        
        convolved_tensor = []
        
        weights = filters(in_channels=1, kernel_size=self.filter_size, filter_type = self.filter_type, filter_params=self.filter_params).to(self.device)
        
        for i in range(in_channels):
                    channel_image = x[:, i:i+1, :, :]
                    channel_convolved = F.conv2d(channel_image, weight= weights.to(self.device), padding=weights.shape[-1] // 2 - 1)
                    convolved_tensor.append(channel_convolved)
            
        
        x = torch.cat(convolved_tensor, dim=1)   
 
        return x    


def initialize_conv_layer(conv_layer, initialization):
    
    init_type = ['kaiming_uniform', 'kaiming_normal', 'orthogonal', 'xavier_uniform', 'xavier_normal', 'uniform','normal']

    assert initialization in init_type, f'invalid initialization type, choose one of {init_type}'
        
    match initialization:
        
        case 'kaiming_uniform':
            torch.nn.init.kaiming_uniform_(conv_layer.weight)
            
        case 'kaiming_normal':
            torch.nn.init.kaiming_normal_(conv_layer.weight)
            
        case 'orthogonal':
            torch.nn.init.orthogonal_(conv_layer.weight) 
            
        case 'xavier_uniform':
            torch.nn.init.xavier_uniform_(conv_layer.weight) 
            
        case 'xavier_normal':
            torch.nn.init.xavier_normal_(conv_layer.weight)  
            
        case 'uniform':
            torch.nn.init.uniform_(conv_layer.weight)     
            
        case 'normal':
            torch.nn.init.normal_(conv_layer.weight)     

    return

def get_kernel_params(kernel_type):
    
    if kernel_type == 'curvature':
        return {'n_ories':12,'n_curves':3,'gau_sizes':(5,),'spatial_fre':[1.2]}
    
    elif kernel_type == 'gabor':
         return {'n_ories':12,'num_scales':3}
        
    elif kernel_type in discrete_wavelets:
        return None

    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")



def get_layer_size(kernel_type, kernel_params):


        if kernel_type == 'curvature':
            return kernel_params['n_ories']*kernel_params['n_curves']*len(kernel_params['gau_sizes']*len(kernel_params['spatial_fre']))*3
        
        elif kernel_type == 'gabor':
            return kernel_params['n_ories']*kernel_params['num_scales']*3
       
        elif kernel_type in discrete_wavelets:
            wavelet_list = [i for i in pywt.wavelist() if kernel_type in i] 
            return len(wavelet_list) * 2 * 3
        
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")