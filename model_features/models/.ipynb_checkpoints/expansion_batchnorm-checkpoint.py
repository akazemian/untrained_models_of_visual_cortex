import torch
from torch import nn

from layer_operations.convolution import WaveletConvolution, initialize_conv_layer
from layer_operations.output import Output
from layer_operations.nonlinearity import NonLinearity
import math

torch.manual_seed(42)
torch.cuda.manual_seed(42)                        
  

    
class Model(nn.Module):
    

    def __init__(self,
                conv1: nn.Module,
                pool1: nn.Module,
                conv2: nn.Module,
                pool2: nn.Module,
                conv3: nn.Module,
                pool3: nn.Module,
                conv4: nn.Module,
                pool4: nn.Module,
                conv5: nn.Module,
                pool5: nn.Module,
                nl: nn.Module,
                gpool: bool,
                last: nn.Module,
                # specified_mean,
                # specified_var
                ):
        
        super(Model, self).__init__()
        
        
        self.conv1 = conv1 
        self.pool1 = pool1
        
        self.conv2 = conv2
        self.pool2 = pool2
        
        self.conv3 = conv3
        self.pool3 = pool3

        self.conv4 = conv4
        self.pool4 = pool4
        
        self.conv5 = conv5
        self.pool5 = pool5
        
        self.nl = nl
        self.gpool = gpool
        self.last = last
        
    
    
    def forward(self, x:nn.Module):         
   
        
        #layer 1 
        x = self.conv1(x)  # conv 
        batch_norm = nn.BatchNorm2d(x.shape[1], device='cuda')
        x = batch_norm(x)
        x = self.nl(x) # non linearity 
        x = self.pool1(x) # anti-aliasing blurpool  
        
        #layer 2
        x = self.conv2(x)  
        batch_norm = nn.BatchNorm2d(x.shape[1], device='cuda')
        x = batch_norm(x)
        x = self.nl(x) # non linearity 
        x = self.pool2(x) 
        
        #layer 3
        x = self.conv3(x)  
        batch_norm = nn.BatchNorm2d(x.shape[1], device='cuda')
        x = batch_norm(x)
        x = self.nl(x) # non linearity 
        x = self.pool3(x) 

        #layer 4
        x = self.conv4(x)  
        batch_norm = nn.BatchNorm2d(x.shape[1], device='cuda')
        x = batch_norm(x)
        x = self.nl(x) # non linearity 
        x = self.pool4(x) 

        #layer 5
        x = self.conv5(x)  
        batch_norm = nn.BatchNorm2d(x.shape[1], device='cuda')
        x = batch_norm(x)
        x = self.nl(x) # non linearity 
        x = self.pool5(x)  
        
        if self.gpool: # global pool
            H = x.shape[-1]
            gmp = nn.AvgPool2d(H) 
            x = gmp(x)
        
        x = self.last(x) # final layer
        
        return x    


    


class Expansion5L:
    def __init__(self, 
                 filters_1_type:str='curvature',
                 filters_1_params:dict = {'n_ories':12,'n_curves':3,'gau_sizes':(5,),'spatial_fre':[1.2]},
                 filters_2:int=1000,
                 filters_3:int=3000,
                 filters_4:int=5000,
                 filters_5:int=30000,
                 init_type:str = 'kaiming_uniform',
                 gpool:bool=False,
                 non_linearity:str='relu',
                 device:str='cuda',
                ):    
        
        self.filters_1_type = filters_1_type
        self.filters_1_params = filters_1_params
        
        match self.filters_1_type:
        
            case 'curvature':
                self.filters_1 = self.filters_1_params['n_ories']*self.filters_1_params['n_curves']*len(self.filters_1_params['gau_sizes']*len(self.filters_1_params['spatial_fre']))*3
            case 'gabor':
                self.filters_1 = self.filters_1_params['n_ories']*self.filters_1_params['num_scales']*3
            case 'random':
                self.filters_1 = self.filters_1_params['filters']
        
        self.filters_2 = filters_2
        self.filters_3 = filters_3
        self.filters_4 = filters_4
        self.filters_5 = filters_5
        self.init_type = init_type
        self.gpool = gpool
        self.non_linearity = non_linearity
        
        self.device = device


    def create_layer(self, in_filters, out_filters, kernel_size, stride=1, pool_kernel=2, pool_stride=None,padding=0):
        conv = nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride= stride, padding = padding, bias=False).to(self.device)
        initialize_conv_layer(conv, self.init_type)
        pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride)
        return conv, pool

    def Build(self):
        
        # layer 1
        if self.filters_1_type != 'random':
            conv1 = WaveletConvolution(filter_size=15, filter_type=self.filters_1_type, filter_params=self.filters_1_params, device = self.device)
            pool1 =  nn.AvgPool2d(kernel_size=2)
        else:
            conv1, pool1 = self.create_layer(3, self.filters_1, (15, 15), 1, 2, padding = math.floor(15 / 2))

        # layer 2 to 5
        conv2, pool2 = self.create_layer(self.filters_1, self.filters_2, (7, 7), 1, 2)
        conv3, pool3 = self.create_layer(self.filters_2, self.filters_3, (5, 5), 1, 2)
        conv4, pool4 = self.create_layer(self.filters_3, self.filters_4, (3, 3), 1, 2)
        conv5, pool5 = self.create_layer(self.filters_4, self.filters_5, (3, 3), 1, 4, 1)

        nl = NonLinearity(self.non_linearity)
        last = Output()

        return Model(
            conv1, pool1, conv2, pool2, conv3, pool3, conv4, pool4, conv5, pool5, nl, self.gpool, last,
        )
