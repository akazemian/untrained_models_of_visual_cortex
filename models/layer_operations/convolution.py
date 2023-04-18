from models.layer_operations.filters import filters
from models.layer_operations.nonlinearity import nonlinearity
from torch import nn
from torch.nn import functional as F
import math
import torch



class StandardConvolution(nn.Module):
    
    """
    Attributes
    ----------
    filter_type  
        The type of filter used for convolution. One of : random, curvature, 1x1
    
    curv_params
        the parametrs used to create the filters. applicable for curvature filters
        
    filter_size 
        The kernel size used in layer. 
    
    out_channels
        the number of filters used for convolution 
    pooling
        the type of pooling used. must be a tuple with the first element being the pooling type (max or avg) and the second the pooling size. Ex: pooling = (max,6)
"""
    
    
    def __init__(self, filter_type:str,
                 curv_params:dict=None,
                 filter_size:int=None,
                 out_channels:int=None,
                 pooling:tuple=None,
                 nonlinearities:list=None):
        
        super().__init__()
        
        self.out_channels = out_channels
        self.filter_type = filter_type
        self.filter_size = filter_size
        self.curv_params = curv_params
        self.pooling = pooling
        self.nonlinearities = nonlinearities
    

    
    
    def extra_repr(self) -> str:
        return 'out_channels={out_channels}, kernel_size={filter_size}, filter_type:{filter_type},pooling={pooling},curv_params:{curv_params}'.format(**self.__dict__)
    
    
    
    def forward(self,x):
            
        
        in_channels = x.shape[1]

        # for RGB input
        if in_channels == 3:
            w = filters(filter_type=self.filter_type,out_channels=self.out_channels,in_channels=1,
                         kernel_size=self.filter_size,curv_params=self.curv_params)
            weight = w.repeat(1,3,1,1)

            
        # grayscale input
        else:
            weight = filters(filter_type=self.filter_type,out_channels=self.out_channels,in_channels=in_channels,
                         kernel_size=self.filter_size,curv_params=self.curv_params)
        
        weight = weight.cuda()
        x =  x.cuda()
        x = F.conv2d(x,weight=weight,padding=math.floor(weight.shape[-1] / 2))

        if self.nonlinearities is not None:
            for operation in self.nonlinearities:
                assert operation in ['zscore', 'norm','relu','gelu','abs'], "nonlinearity doesnt match any available operation"
                nl = nonlinearity(operation=operation)
                x = nl(x)
                print(operation)
                
         
        if self.pooling is not None:
            assert self.pooling[0] in ['max','avg'], "pooling operation should be max or avg"
            if self.pooling[0] == 'max':
                mp = nn.MaxPool2d(self.pooling[1])
                x = mp(x)
            else:
                mp = nn.AvgPool2d(self.pooling[1])
                x = mp(x)   

        return x
        

    
    
    
class RandomProjections(nn.Module): 
    
    def __init__(self, out_channels,max_pool=None):
        super().__init__()

        self.out_channels = out_channels
        self.pooling = max_pool
    
    def extra_repr(self) -> str:
        return 'out_channels={out_channels},max_pool={max_pool}'.format(**self.__dict__)
    
    
    def forward(self, x):
        
        in_channels = x.shape[1] 
        weight = filters(filter_type='1x1',out_channels=self.out_channels,in_channels=in_channels)
 
        weight = weight.cuda()
        x = torch.Tensor(x)
        x =  x.cuda()
        
        x = F.conv2d(x,weight=weight,padding=0)
        
         
        if self.pooling == None:
            pass
        else:
            assert self.pooling[0] in ['max','avg'], "pooling operation should be one of max or avg"
            if self.pooling[0] == 'max':
                mp = nn.MaxPool2d(self.pooling[1])
                x = mp(x)
            else:
                mp = nn.AvgPool2d(self.pooling[1])
                x = mp(x)  
        return x
        
        
        