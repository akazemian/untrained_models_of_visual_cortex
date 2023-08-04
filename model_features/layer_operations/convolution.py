from model_features.layer_operations.preset_filters import filters
from torch.nn import functional as F
import math
import torch
torch.manual_seed(42)
from torch import nn


class Convolution(nn.Module):
    
    """
    Attributes
    ----------
    filter_type  
        The type of filter used for convolution. One of [curvature, gabor]
    
    curv_params
        the parametrs used to create the filters, applicable for curvature filters
        
    filter_size 
        The kernel size used in layer. 
    

    """
    
    
    def __init__(self, 
                 filter_params:dict=None,
                 filter_size:int=None,
                 device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
                ):
                
        super().__init__()
        

        self.filter_size = filter_size
        self.filter_params = filter_params
        self.device = device
    
    def extra_repr(self) -> str:
        return 'kernel_size={filter_size}, filter_params:{filter_params}'.format(**self.__dict__)
    
    
    
    def forward(self,x):
            
        
        x =  x.to(self.device)
        in_channels = x.shape[1]
        
        weights = filters(in_channels=1, kernel_size=self.filter_size, filter_params=self.filter_params)
        weights = weights.to(self.device)
        
        
        # for RGB input (the preset L1 filters are repeated across the 3 channels)
            
        convolved_tensor = []
        for i in range(in_channels):
            channel_image = x[:, i:i+1, :, :]
            channel_convolved = F.conv2d(channel_image, weight= weights, padding=math.floor(weights.shape[-1] / 2))
            convolved_tensor.append(channel_convolved)
        x = torch.cat(convolved_tensor, dim=1)    
    

        return x
    




        
def initialize_conv_layer(conv_layer, initialization):
    
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
            
        case _:
            raise ValueError(f"Unsupported initialization type: {initialization}.")      
        
        