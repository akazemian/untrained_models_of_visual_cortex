from model_features.layer_operations.convolution import Convolution
from model_features.layer_operations.output import Output
from model_features.layer_operations.blurpool import BlurPool
from model_features.layer_operations.nonlinearity import NonLinearity
import torch
from torch import nn
import os
import sys
ROOT = os.getenv('BONNER_ROOT_PATH')
from model_features.models.expansion_3_layers import Model

    
    
class FullyRandom:
        
    
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
    
    device
        device used for the first convolution layer, this layer may often npt fit on gpu, in which case device should be set to cpu
    """
    



    def __init__(self, 
                 filters_1:int=36,
                 filters_2:int=2000,
                 filters_3:int=10000,
                 batches_3:int = 1,
                 init_type:str = 'kaiming_uniform',
                 bpool_filter_size:int=4,
                 gpool:bool=True):    
        
        self.filters_1 = filters_1
        self.filters_2 = filters_2
        self.filters_3 = filters_3
        self.batches_3 = batches_3
        self.init_type = init_type
        self.bpool_filter_size = bpool_filter_size
        self.gpool = gpool
    
    
    
    def Build(self):        
        
        # layer 1
        conv1 = nn.Conv2d(3, self.filters_1, kernel_size=(15, 15)) 
        bpool1 = BlurPool(self.filters_1, filt_size=self.bpool_filter_size, stride=2)
        pool1 = nn.AvgPool2d(kernel_size=3)
        

        
        # layer 2
        conv2 = nn.Conv2d(self.filters_1, self.filters_2, kernel_size=(9, 9))
        bpool2 = BlurPool(self.filters_2, filt_size=self.bpool_filter_size, stride=2)
        pool2 = nn.AvgPool2d(kernel_size=2,stride=1)
        
        
        # layer 3
        conv3 = nn.Conv2d(self.filters_2, self.filters_3, kernel_size=(7,7))
        bpool3 = BlurPool(self.filters_3*self.batches_3, filt_size=self.bpool_filter_size, stride=2)
        pool3 = nn.AvgPool2d(kernel_size=2,stride=1)

        # non lineairy function
        nl = NonLinearity('relu')
        
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
                last = last,
        )
    