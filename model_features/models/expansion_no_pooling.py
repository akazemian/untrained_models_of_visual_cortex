import torch
from torch import nn

from layer_operations.convolution import WaveletConvolution, initialize_conv_layer
from layer_operations.output import Output
from layer_operations.nonlinearity import NonLinearity
torch.manual_seed(42)
torch.cuda.manual_seed(42)                        
  

    
class Model(nn.Module):
    

    def __init__(self,
                conv1: nn.Module,
                conv2: nn.Module,
                conv3: nn.Module,
                conv4: nn.Module,
                conv5: nn.Module,
                nl: nn.Module,
                gpool: bool,
                last: nn.Module,
                ):
        
        super(Model, self).__init__()
        
        
        self.conv1 = conv1 
        self.conv2 = conv2
        self.conv3 = conv3
        self.conv4 = conv4
        self.conv5 = conv5
        
        self.nl = nl
        self.gpool = gpool
        self.last = last
        
        
    def forward(self, x:nn.Module):         
   
        #layer 1 
        x = self.conv1(x)  # conv 
        print(x.shape)
        x = self.nl(x) # non linearity 
        
        #layer 2
        x = self.conv2(x)  
        print(x.shape)
        x = self.nl(x) 
            
        #layer 3
        x = self.conv3(x)  
        print(x.shape)
        x = self.nl(x) 

        #layer 4
        x = self.conv4(x)  
        print(x.shape)
        x = self.nl(x) 
        
        #layer 5
        x = self.conv5(x)  
        print(x.shape)
        x = self.nl(x) 
        
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
        self.gpool = gpool
        self.non_linearity = non_linearity
        
        self.device = device
        

    def create_layer(self, in_filters, out_filters, kernel_size, stride=2):
        conv = nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride, bias=False).to(self.device)
        initialize_conv_layer(conv, self.init_type)
        return conv

    def Build(self):
        # layer 1
        conv1 = WaveletConvolution(filter_size=15, filter_params=self.filter_params, device = self.device)

        # layer 2 to 5
        conv2 = self.create_layer(self.filters_1, self.filters_2, (7, 7),1)
        conv3 = self.create_layer(self.filters_2, self.filters_3, (5, 5))
        conv4 = self.create_layer(self.filters_3, self.filters_4, (3, 3))
        conv5 = self.create_layer(self.filters_4, self.filters_5, (3, 3))

        nl = NonLinearity(self.non_linearity)
        last = Output()

        return Model(
            conv1, conv2, conv3, conv4, conv5, nl, self.gpool, last
        )


    
