import torch
from torch import nn

from layer_operations.convolution import WaveletConvolution, initialize_conv_layer
from layer_operations.output import Output
from layer_operations.blurpool import BlurPool
from layer_operations.nonlinearity import NonLinearity
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
                randproj5: nn.Module,
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
        self.randproj5 = randproj5
        
        self.gpool = gpool
        self.last = last
        
        
    def forward(self, x:nn.Module):         
   
        #layer 1 
        x = self.conv1(x)  # conv 
        x = self.bpool1(x) # anti-aliasing blurpool               

            
        #layer 2
        x = self.conv2(x)  
        x = self.bpool2(x) 
            
        #layer 3
        x = self.conv3(x)  
        x = self.bpool3(x) 

        #layer 4
        x = self.conv4(x)  
        x = self.bpool4(x) 
        
        #layer 5
        x = self.conv5(x)  
        x = self.pool5(x)  
        
        if self.gpool: # global pool
            H = x.shape[-1]
            gmp = nn.AvgPool2d(H) 
            x = gmp(x)

        
        x = self.last(x) # final layer
        
        return x    


    
    
  

    
class Expansion5LLinear:
        
    

    def __init__(self, 
                 filter_params:dict = {'type':'curvature','n_ories':12,'n_curves':3,'gau_sizes':(5,),'spatial_fre':[1.2]},
                 filters_2:int=1000,
                 filters_3:int=5000,
                 filters_4:int=5000,
                 filters_5:int=30000,
                 init_type:str = 'kaiming_uniform',
                 bpool_filter_size:int=4,
                 gpool:bool=False,
                device='cuda'):    
        
        
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
        self.device = device
    
    
    
    def Build(self):        
        
        # layer 1
        conv1 = WaveletConvolution(filter_size=15,filter_params=self.filter_params, device = self.device)     
        bpool1 = BlurPool(self.filters_1, filt_size=self.bpool_filter_size, stride=2)
        pool1 = nn.AvgPool2d(kernel_size=2)

        
        # layer 2
        conv2 = nn.Conv2d(self.filters_1, self.filters_2, kernel_size=(7,7), bias=False)
        initialize_conv_layer(conv2, self.init_type)
        bpool2 = BlurPool(self.filters_2, filt_size=self.bpool_filter_size, stride=2)
        pool2 = nn.AvgPool2d(kernel_size=2, stride=1)
        
        
        # layer 3
        conv3 = nn.Conv2d(self.filters_2, self.filters_3, kernel_size=(5,5), bias=False)
        initialize_conv_layer(conv3, self.init_type)
        bpool3 = BlurPool(self.filters_3, filt_size=self.bpool_filter_size, stride=2)
        pool3 = nn.AvgPool2d(kernel_size=2, stride=1)

        # layer 4
        conv4 = nn.Conv2d(self.filters_3, self.filters_4, kernel_size=(3,3), bias=False)
        initialize_conv_layer(conv4, self.init_type)
        bpool4 = BlurPool(self.filters_4, filt_size=self.bpool_filter_size, stride=2)
        pool4 = nn.AvgPool2d(kernel_size=2, stride=1)
        
        # layer 5
        conv5 = nn.Conv2d(self.filters_4, self.filters_5, kernel_size=(3,3), bias=False)
        initialize_conv_layer(conv5, self.init_type)
        bpool5 = BlurPool(self.filters_5, filt_size=self.bpool_filter_size, stride=2)
        pool5 = nn.AvgPool2d(kernel_size=5, stride=1)
        randproj5 = nn.Conv2d(self.filters_5, 3000, kernel_size=(1,1), bias=False)
        
        #readout layer
        last = Output()

        
        return Model5L(
                conv1 = conv1,
                bpool1 = bpool1,
                pool1 = pool1,
                conv2 = conv2,
                bpool2 = bpool2,
                pool2 = pool2,
                conv3 = conv3,
                bpool3 = bpool3,
                pool3 = pool3,
                conv4 = conv4,
                bpool4 = bpool4,
                pool4 = pool4,
                conv5 = conv5,
                bpool5 = bpool5,
                pool5 = pool5,
            randproj5 = randproj5,
                gpool = self.gpool,
                last = last,
        )
    



