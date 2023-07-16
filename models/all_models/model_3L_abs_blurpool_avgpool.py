from models.layer_operations.convolution import Convolution
from models.layer_operations.output import Output
from models.layer_operations.blurpool import BlurPool
from models.layer_operations.nonlinearity import NonLinearity
import torch
from torch import nn
                         

class Conv3Module(nn.Module):
    def __init__(self, conv_layer, non_linear_activation):
        super(Conv3Module, self).__init__()
        self.conv_layer = conv_layer
        self.non_linear_activation = non_linear_activation

    def forward(self, x):
        x = self.non_linear_activation(self.conv_layer(x))
        return x
            

        
class Model(nn.Module):
    
    
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
                print_shape: bool = True
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
        self.print_shape = print_shape
        
        
    def forward(self, x:nn.Module): 
        
        
        x = x.to('cuda')
        print(x.shape)
        
        #layer 1 
        x = self.conv1(x)  # conv 
        print('c1',x.shape)  
        x = self.nl(x) # non linearity 
        x = self.bpool1(x) # anti-aliasing blurpool                
        x = self.pool1(x) # pool
        print('p1',x.shape)  

            
        #layer 2
        x = self.conv2(x)  
        print('c2',x.shape)  
        x = self.nl(x) 
        x = self.bpool2(x) 
        x = self.pool2(x)    
        print('p2',x.shape)   
         

        x_repeated = x.repeat(self.batches_3, 1, 1, 1)
        conv_3 = self.nl(self.conv3(x_repeated))
        x = torch.cat(torch.chunk(conv_3, self.batches_3, dim=0), dim=1)
        print('c3',x.shape)  


        x = self.bpool3(x)
        x = self.pool3(x)
        print('p3',x.shape)

        
        if self.gpool: # global pool
            H = x.shape[-1]
            gmp = nn.AvgPool2d(H) 
            x = gmp(x)

        
        x = self.last(x) # final layer
        print('final',x.shape)

        
        return x    


    
    
  

    
class ExpansionModel:
        
    
    """
    Attributes
    ----------
    curv_params  
        pre-set curvature filter parameters 
    
    filters_2
        number of random filters in layer 2

        
    filters_3 
        number of random filters in layer 3
    
    batches_3
        number of batches used for layer 3 convolution. Used in case of memory issues 
        to perform convolution in batches. The number of output channles is equal to 
        filters_3 x batches_3
    
    bpool_filter_size
        kernel size for the anti aliasing operation (blurpool)

    gpool:
        whether global pooling is performed on the output of layer 3 
    """
    



    def __init__(self, 
                 curv_params:dict = {'n_ories':12,'n_curves':3,'gau_sizes':(5,),'spatial_fre':[1.2]},
                 filters_2:int=1000,
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
        bpool1 = BlurPool(self.filters_1*3, filt_size=self.bpool_filter_size, stride=2)
        pool1 = nn.AvgPool2d(kernel_size=3)

        
        # layer 2
        conv2 = nn.Conv2d(self.filters_1*3, self.filters_2, kernel_size=(9, 9), bias=False)
        initialize_conv_layer(conv2, self.init_type)
        bpool2 = BlurPool(self.filters_2, filt_size=self.bpool_filter_size, stride=2)
        pool2 = nn.AvgPool2d(kernel_size=2, stride=1)
        
        
        # layer 3
        conv3 = nn.Conv2d(self.filters_2, self.filters_3, kernel_size=(7,7), bias=False)
        initialize_conv_layer(conv3, self.init_type)
        bpool3 = BlurPool(self.filters_3*self.batches_3, filt_size=self.bpool_filter_size, stride=2)
        pool3 = nn.AvgPool2d(kernel_size=2, stride=1)

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