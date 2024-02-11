import torch
from torch import nn

from model_features.layer_operations.convolution import Convolution, initialize_conv_layer, NonSharedConv2d
from model_features.layer_operations.output import Output
from model_features.layer_operations.blurpool import BlurPool
from model_features.layer_operations.nonlinearity import NonLinearity
torch.manual_seed(42)
torch.cuda.manual_seed(42)                        
  

    
class Model5L(nn.Module):
    

    def __init__(self,
                conv1: nn.Module,
                bpool1: nn.Module,
                pool1: nn.Module,
                conv2: nn.Module,
                bpool2: nn.Module,
                pool2: nn.Module,
                conv3: nn.Module,
                bpool3: nn.Module,
                pool3: nn.Module,
                conv4: nn.Module,
                bpool4: nn.Module,
                pool4: nn.Module,
                conv5: nn.Module,
                bpool5: nn.Module,
                pool5: nn.Module,
                nl: nn.Module,
                gpool: bool,
                last: nn.Module,
                ):
        
        super(Model5L, self).__init__()
        
        
        self.conv1 = conv1 
        self.bpool1 = bpool1 
        self.pool1 = pool1
        
        self.conv2 = conv2
        self.bpool2 = bpool2
        self.pool2 = pool2
        
        self.conv3 = conv3
        self.bpool3 = bpool3
        self.pool3 = pool3

        self.conv4 = conv4
        self.bpool4 = bpool4
        self.pool4 = pool4
        
        self.conv5 = conv5
        self.bpool5 = bpool5
        self.pool5 = pool5
        
        self.nl = nl
        self.gpool = gpool
        self.last = last
        
        
    def forward(self, x:nn.Module):         
   
        #layer 1 
        x = self.conv1(x)  # conv 
        x = self.nl(x) # non linearity 
        x = self.bpool1(x) # anti-aliasing blurpool               
            
        #layer 2
        x = self.conv2(x)  
        x = self.nl(x) 
        x = self.bpool2(x) 
            
        #layer 3
        x = self.conv3(x)  
        x = self.nl(x) 
        x = self.bpool3(x) 

        #layer 4
        x = self.conv4(x)  
        x = self.nl(x) 
        x = self.bpool4(x) 
        
        #layer 5
        x = self.conv5(x)  
        x = self.nl(x) 
        x = self.pool5(x)  
        
        if self.gpool: # global pool
            H = x.shape[-1]
            gmp = nn.AvgPool2d(H) 
            x = gmp(x)

        
        x = self.last(x) # final layer
        
        return x    


    


class Expansion5L:
    def __init__(self, 
                 filter_params:dict = {'type':'curvature','n_ories':12,'n_curves':3,'gau_sizes':(5,),'spatial_fre':[1.2]},
                 filters_2:int=1000,
                 filters_3:int=3000,
                 filters_4:int=5000,
                 filters_5:int=30000,
                 init_type:str = 'kaiming_uniform',
                 bpool_filter_size:int=4,
                 gpool:bool=False,
                 non_linearity:str='relu',
                device:str='cuda'):    
        
        
        self.filter_params = filter_params
        
        if self.filter_params['type'] == 'curvature':
            self.filters_1 = self.filter_params['n_ories']*self.filter_params['n_curves']*len(self.filter_params['gau_sizes']*len(self.filter_params['spatial_fre']))*3
        else:
            self.filters_1 = self.filter_params['n_ories']*self.filter_params['num_scales']*3
        
        self.filters_2 = filters_2
        self.filters_3 = filters_3
        self.filters_4 = filters_4
        self.filters_5 = filters_5
        self.init_type = init_type
        self.bpool_filter_size = bpool_filter_size
        self.gpool = gpool
        self.non_linearity = non_linearity
        
        self.device = device
        

    def create_layer(self, in_filters, out_filters, kernel_size, stride, pool_kernel, pool_stride):
        conv = nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, bias=False)
        initialize_conv_layer(conv, self.init_type)
        bpool = BlurPool(out_filters, filt_size=self.bpool_filter_size, stride=stride)
        pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride)
        return conv, bpool, pool

    def Build(self):
        # layer 1
        conv1 = Convolution(filter_size=15, filter_params=self.filter_params, device = self.device)
        bpool1, pool1 = BlurPool(self.filters_1, filt_size=self.bpool_filter_size, stride=2), nn.AvgPool2d(kernel_size=2)

        # layer 2 to 5
        conv2, bpool2, pool2 = self.create_layer(self.filters_1, self.filters_2, (7, 7), 2, 2, 1)
        conv3, bpool3, pool3 = self.create_layer(self.filters_2, self.filters_3, (5, 5), 2, 2, 1)
        conv4, bpool4, pool4 = self.create_layer(self.filters_3, self.filters_4, (3, 3), 2, 2, 1)
        conv5, bpool5, pool5 = self.create_layer(self.filters_4, self.filters_5, (3, 3), 2, 5, 1)

        nl = NonLinearity(self.non_linearity)
        last = Output()

        return Model5L(
            conv1, bpool1, pool1, conv2, bpool2, pool2, conv3, bpool3, pool3, conv4, bpool4, pool4, conv5, bpool5, pool5, nl, self.gpool, last
        )





class ExpansionNoWeightShare:
    def __init__(self, 
                 filter_params:dict = {'type':'curvature','n_ories':12,'n_curves':3,'gau_sizes':(5,),'spatial_fre':[1.2]},
                 filters_2:int=1000,
                 filters_3:int=3000,
                 filters_4:int=5000,
                 filters_5:int=30000,
                 init_type:str = 'kaiming_uniform',
                 bpool_filter_size:int=4,
                 gpool:bool=False,
                 non_linearity:str='relu'):    
        
        
        self.filter_params = filter_params
        
        if self.filter_params['type'] == 'curvature':
            self.filters_1 = self.filter_params['n_ories']*self.filter_params['n_curves']*len(self.filter_params['gau_sizes']*len(self.filter_params['spatial_fre']))*3
        else:
            self.filters_1 = self.filter_params['n_ories']*self.filter_params['num_scales']*3
        
        self.filters_2 = filters_2
        self.filters_3 = filters_3
        self.filters_4 = filters_4
        self.filters_5 = filters_5
        self.init_type = init_type
        self.bpool_filter_size = bpool_filter_size
        self.gpool = gpool
        self.non_linearity = non_linearity
        

    def create_layer(self, in_filters, out_filters, kernel_size, stride, pool_kernel, pool_stride):
        
        conv = NonSharedConv2d(in_filters, out_filters, kernel_size, stride=1, padding=0)
        bpool = BlurPool(out_filters, filt_size=self.bpool_filter_size, stride=stride)
        pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride)
        return conv, bpool, pool

    def Build(self):
        # layer 1
        conv1 = Convolution(filter_size=15, filter_params=self.filter_params)
        bpool1, pool1 = BlurPool(self.filters_1, filt_size=self.bpool_filter_size, stride=2), nn.AvgPool2d(kernel_size=2)

        # layer 2 to 5
        conv2, bpool2, pool2 = self.create_layer(self.filters_1, self.filters_2, (7, 7), 2, 2, 1)
        conv3, bpool3, pool3 = self.create_layer(self.filters_2, self.filters_3, (5, 5), 2, 2, 1)
        conv4, bpool4, pool4 = self.create_layer(self.filters_3, self.filters_4, (3, 3), 2, 2, 1)
        conv5, bpool5, pool5 = self.create_layer(self.filters_4, self.filters_5, (3, 3), 2, 5, 1)

        nl = NonLinearity(self.non_linearity)
        last = Output()

        return Model5L(
            conv1, bpool1, pool1, conv2, bpool2, pool2, conv3, bpool3, pool3, conv4, bpool4, pool4, conv5, bpool5, pool5, nl, self.gpool, last
        )

    
    
    
