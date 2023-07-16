from models.layer_operations.preset_filters import filters
from torch import nn
from torch.nn import functional as F
import math
import torch


class Convolution(nn.Module):
    
    """
    Attributes
    ----------
    filter_type  
        The type of filter used for convolution. One of : random, curvature, 1x1
    
    curv_params
        the parametrs used to create the filters, applicable for curvature filters
        
    filter_size 
        The kernel size used in layer. 
    

    """
    
    
    def __init__(self, filter_type:str,
                 curv_params:dict=None,
                 filter_size:int=None
                ):
                
        super().__init__()
        

        self.filter_type = filter_type
        self.filter_size = filter_size
        self.curv_params = curv_params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    
    
    def extra_repr(self) -> str:
        return 'kernel_size={filter_size}, filter_type:{filter_type},curv_params:{curv_params}'.format(**self.__dict__)
    
    
    
    def forward(self,x):
            
        
        in_channels = x.shape[1]
        
        weights = filters(filter_type=self.filter_type,in_channels=1,
                     kernel_size=self.filter_size,curv_params=self.curv_params)
        weights = weights.to(self.device)
        print('weight',weights.shape)
        x =  x.to(self.device)
        
        # for RGB input (the preset L1 filters are repeated across the 3 channels)
        if in_channels == 3: 
            
            convolved_tensor = []
            for i in range(3):
                channel_image = x[:, i:i+1, :, :]
                channel_convolved = F.conv2d(channel_image, weight= weights, padding=math.floor(weights.shape[-1] / 2))
                convolved_tensor.append(channel_convolved)

            # Combine the convolved channels
            x = torch.cat(convolved_tensor, dim=1)
    
    
        # for grayscale input
        else: 
            print('image shape:',x.shape)
            x = F.conv2d(x,weight=weights,padding=math.floor(weights.shape[-1] / 2))

        return x
    




        
        
        
        