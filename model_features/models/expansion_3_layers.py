import torch
from torch import nn

from model_features.layer_operations.convolution import Convolution, initialize_conv_layer
from model_features.layer_operations.output import Output
from model_features.layer_operations.blurpool import BlurPool
from model_features.layer_operations.nonlinearity import NonLinearity
torch.manual_seed(42)
torch.cuda.manual_seed(42)                        

        
class Model(nn.Module):
    
    
    """
    Neural network architecture consisting of three layers.
    
    Attributes:
    -----------
    conv1 : nn.Module
        Convolutional layer for the first layer.
    bpool1 : nn.Module
        BlurPool (anti-aliasing) layer for the first layer.
    pool1 : nn.Module
        Average pooling layer for the first layer.
    conv2 : nn.Module
        Convolutional layer for the second layer.
    bpool2 : nn.Module
        BlurPool layer for the second layer.
    pool2 : nn.Module
        Average pooling layer for the second layer.
    conv3 : nn.Module
        Convolutional layer for the third layer.
    bpool3 : nn.Module
        BlurPool layer for the third layer.
    pool3 : nn.Module
        Average pooling layer for the third layer.
    batches_3 : int
        Number of batches used in the third layer's convolution.
    nl : nn.Module
        Non-linearity module.
    gpool : bool
        If true, global pooling is applied after the third layer.
    last : nn.Module
        Final output layer.
    """

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
                batches_3: int,
                nl: nn.Module,
                gpool: bool,
                last: nn.Module,
                ):
        
        super(Model, self).__init__()
        
        
        self.conv1 = conv1 
        self.bpool1 = bpool1 
        self.pool1 = pool1
        
        self.conv2 = conv2
        self.bpool2 = bpool2
        self.pool2 = pool2
        
        self.conv3 = conv3
        self.bpool3 = bpool3
        self.pool3 = pool3
        self.batches_3 = batches_3
        
        self.nl = nl
        self.gpool = gpool
        self.last = last
        
        
    def forward(self, x:nn.Module):         
   
        #layer 1 
        x = self.conv1(x)  # conv 
        x = self.nl(x) # non linearity 
        x = self.bpool1(x) # anti-aliasing blurpool               
        x = self.pool1(x) # pool

            
        #layer 2
        x = self.conv2(x)  
        x = self.nl(x) 
        x = self.bpool2(x) 
        x = self.pool2(x)    
         
            
        #layer 3
        x_repeated = x.repeat(self.batches_3, 1, 1, 1)
        conv_3 = self.nl(self.conv3(x_repeated))
        x = torch.cat(torch.chunk(conv_3, self.batches_3, dim=0), dim=1)
        x = self.bpool3(x)
        x = self.pool3(x)

        
        if self.gpool: # global pool
            H = x.shape[-1]
            gmp = nn.AvgPool2d(H) 
            x = gmp(x)

        
        x = self.last(x) # final layer
        
        return x    


    
    
  

    
class Expansion:
        
    """
    Builds a Model instance based on various filter parameters and configurations.
    
    Attributes:
    -----------
    filter_params : dict
        Contains parameters for preset filter configuration.
    filters_2 : int
        Number of random filters for the second layer.
    filters_3 : int
        Number of random filters for the third layer.
    batches_3 : int
        Number of batches used for layer 3 convolution.
    init_type : str
        Type of weight initialization for convolution layers.
    bpool_filter_size : int
        Kernel size for blurpool operation.
    gpool : bool
        If true, global pooling is applied on the output of layer 3.
    non_linearity : str
        Type of non-linear activation function.
    """
    

    def __init__(self, 
                 filter_params:dict = {'type':'curvature','n_ories':12,'n_curves':3,'gau_sizes':(5,),'spatial_fre':[1.2]},
                 filters_2:int=1000,
                 filters_3:int=10000,
                 batches_3:int = 1,
                 init_type:str = 'kaiming_uniform',
                 bpool_filter_size:int=4,
                 gpool:bool=True,
                 non_linearity:str='relu'):    
        
        
        self.filter_params = filter_params
        
        if self.filter_params['type'] == 'curvature':
            self.filters_1 = self.filter_params['n_ories']*self.filter_params['n_curves']*len(self.filter_params['gau_sizes']*len(self.filter_params['spatial_fre']))*3
        else:
            self.filters_1 = self.filter_params['n_ories']*self.filter_params['num_scales']*3
        
        self.filters_2 = filters_2
        self.filters_3 = filters_3
        self.batches_3 = batches_3
        self.init_type = init_type
        self.bpool_filter_size = bpool_filter_size
        self.gpool = gpool
        self.non_linearity = non_linearity
    
    
    
    def Build(self):        
        
        # layer 1
        conv1 = Convolution(filter_size=15,filter_params=self.filter_params)     
        bpool1 = BlurPool(self.filters_1, filt_size=self.bpool_filter_size, stride=2)
        pool1 = nn.AvgPool2d(kernel_size=3)

        
        # layer 2
        conv2 = nn.Conv2d(self.filters_1, self.filters_2, kernel_size=(9, 9), bias=False)
        initialize_conv_layer(conv2, self.init_type)
        bpool2 = BlurPool(self.filters_2, filt_size=self.bpool_filter_size, stride=2)
        pool2 = nn.AvgPool2d(kernel_size=2, stride=1)
        
        
        # layer 3
        conv3 = nn.Conv2d(self.filters_2, self.filters_3, kernel_size=(7,7), bias=False)
        initialize_conv_layer(conv3, self.init_type)
        bpool3 = BlurPool(self.filters_3*self.batches_3, filt_size=self.bpool_filter_size, stride=2)
        pool3 = nn.AvgPool2d(kernel_size=2, stride=1)

        # non lineairy function
        nl = NonLinearity(self.non_linearity)
        
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
    



