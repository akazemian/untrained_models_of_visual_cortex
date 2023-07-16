from models.layer_operations.convolution import Convolution
from models.layer_operations.output import Output
from models.layer_operations.blurpool import BlurPool
from models.layer_operations.nonlinearity import NonLinearity
import torch
from torch import nn
                         


class Model(nn.Module):
    
    
    def __init__(self,
                conv1: nn.Module,
                bpool1: nn.Module,
                pool1: nn.Module,
                batches_1: int,
                nl: nn.Module,
                gpool: bool,
                last: nn.Module,
                print_shape: bool = True
                ):
        
        super(Model, self).__init__()
        
        
        self.conv1 = conv1 
        self.bpool1 = bpool1 
        self.pool1 = pool1
        self.batches_1 = batches_1
        
        self.nl = nl
        self.gpool = gpool
        self.last = last
        self.print_shape = print_shape
        
        
    def forward(self, x:nn.Module): 
               
            
            
        #layer 1
        conv_1 = []
        for i in range(self.batches_1):
            conv_1.append(self.conv1(x))  
        x = torch.cat(conv_1) 
        x = self.bpool1(x) 
        x = self.pool1(x) 

        
        if self.gpool: # global pool
            H = x.shape[-1]
            gmp = nn.AvgPool2d(H) 
            x = gmp(x)

        
        x = self.last(x) # final layer

        
        return x    


    
    
  

    
class ExpansionModel1L:
        
    
    """
    Attributes
    ----------
    curv_params  
        pre-set curvature filter parameters 
    
    batches_1
        number of batches used for layer 1 convolution. Used in case of memory issues 
        to perform convolution in batches. The number of output channles is equal to 
        filters_1 x batches_1
    
    bpool_filter_size
        kernel size for the anti aliasing operation (blurpool)

    gpool:
        whether global pooling is performed on the output of layer 1 
    """
    



    def __init__(self, 
                 curv_params:dict = {'n_ories':12,'n_curves':3,'gau_sizes':(5,),'spatial_fre':[1.2]},
                 filters_2:int=2000,
                 filters_3:int=10000,
                 batches_3:int = 1,
                 init_type:str = 'kaiming_uniform',
                 bpool_filter_size:int=4,
                 gpool:bool=True):    
        
        self.curv_params = curv_params
        self.filters_1 = self.curv_params['n_ories']*self.curv_params['n_curves']*len(self.curv_params['gau_sizes']*len(self.curv_params['spatial_fre']))
        self.filters_2 = filters_2
        self.filters_3 = filters_3
        self.batches_3 = batches_3
        self.init_type = init_type
        self.bpool_filter_size = bpool_filter_size
        self.gpool = gpool
    
    
    
    def Build(self):        
        
        # layer 1
        conv1 = Convolution(filter_size=15,filter_type='curvature',curv_params=self.curv_params)     
        bpool1 = BlurPool(36, filt_size=self.bpool_filter_size, stride=2)
        pool1 = nn.AvgPool2d(kernel_size=3)

        
        # layer 2
        conv2 = nn.Conv2d(36, self.filters_2, kernel_size=(9, 9), bias=False)
        initialize_conv_layer(conv2, self.init_type)
        bpool2 = BlurPool(self.filters_2, filt_size=self.bpool_filter_size, stride=2)
        pool2 = nn.AvgPool2d(kernel_size=2,stride=1)
        
        
        # layer 3
        conv3 = nn.Conv2d(self.filters_2, self.filters_3, kernel_size=(7,7), bias=False)
        initialize_conv_layer(conv3, self.init_type)
        bpool3 = BlurPool(self.filters_3*self.batches_3, filt_size=self.bpool_filter_size, stride=2)
        pool3 = nn.AvgPool2d(kernel_size=2,stride=1)

        # non lineairy function
        nl = NonLinearity('abs')
        
        #readout layer
        last = Output()
        

        
        return Model(
                conv1 = conv1,
                bpool1 = bpool1,
                pool1 = pool1,
                conv2 = conv2,
                bpool2 = bpool2,
                pool2 = pool2,
                conv3 = conv3,
                bpool3 = bpool3,
                pool3 = pool3,
                batches_3 = self.batches_3,
                nl = nl,
                gpool = self.gpool,
                last = last
        )
    



def initialize_conv_layer(conv_layer, initialization):
    if initialization == 'kaiming_uniform':
        nn.init.kaiming_uniform_(conv_layer.weight)
    elif initialization == 'kaiming_normal':
        nn.init.kaiming_normal_(conv_layer.weight)
    elif initialization == 'orthogonal':
        nn.init.orthogonal_(conv_layer.weight) 
    elif initialization == 'xavier_uniform':
        nn.init.xavier_uniform_(conv_layer.weight)      
    elif initialization == 'xavier_normal':
        nn.init.xavier_normal_(conv_layer.weight)     
    elif initialization == 'uniform':
        nn.init.uniform_(conv_layer.weight)      
    elif initialization == 'normal':
        nn.init.normal_(conv_layer.weight)      
    else:
        raise ValueError(f"Unsupported initialization type: {initialization}.")