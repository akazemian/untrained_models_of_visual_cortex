from models.layer_operations.convolution import StandardConvolution
from models.layer_operations.output import Output
from models.layer_operations.nonlinearity import nonlinearity
from torchvision.transforms import GaussianBlur
from models.layer_operations.convolution import *
from models.layer_operations.output import Output
import torch
from torch import nn
                         


class Model(nn.Module):
    
    
    def __init__(self,
                c1: nn.Module,
                gb1:nn.Module,
                mp1: nn.Module,
                c2: nn.Module,
                gb2:nn.Module,
                mp2: nn.Module,
                c3: nn.Module,
                gb3:nn.Module,
                mp3: nn.Module,
                batches_3: int,
                nl1: nn.Module,
                last: nn.Module,
                print_shape: bool = True
                ):
        
        super(Model, self).__init__()
        
        
        self.c1 = c1 
        self.gb1 = gb1
        self.mp1 = mp1
        
        self.c2 = c2
        self.gb2 = gb2
        self.mp2 = mp2
        
        self.c3 = c3
        self.gb3 = gb3
        self.mp3 = mp3
        self.batches_3 = batches_3
        
        self.nl1 = nl1
        
        self.last = last
        self.print_shape = print_shape
        
        
    def forward(self, x:nn.Module):
                
        
        #conv layer 1
        x = self.c1(x)
        if self.print_shape:
            print('conv1', x.shape)
            
        
        
        x = self.nl1(x)
        if self.print_shape:
            print('non lin', x.shape)   
            
        # x = self.gb1(x)
        
        x = self.mp1(x)
        if self.print_shape:
            print('mp1', x.shape)
    
            
            
        #conv layer 2
        x = self.c2(x)
        if self.print_shape:
            print('conv2', x.shape)        
        
         
               
        x = self.nl1(x)
        if self.print_shape:
            print('non lin', x.shape)   
            
        x = self.gb2(x)   
        
        x = self.mp2(x)
        if self.print_shape:
            print('mp2', x.shape)
            

            
        #conv layer 3
        conv_3 = []
        for i in range(self.batches_3):
            conv_3.append(self.c3(x)) 
        x = torch.cat(conv_3,dim=1)
        if self.print_shape:
            print('conv3', x.shape)
            
        
        
        x = self.nl1(x)
        if self.print_shape:
            print('non lin', x.shape)   
            
        #x = self.gb3(x)
        
        # x = self.mp3(x)
        # if self.print_shape:
        #     print('mp3', x.shape)
        
 
            
        x = self.last(x)
        if self.print_shape:
            print('output', x.shape)
        
        return x    



  
class EngModel3LAbsGF:
    
    """
    Used to Initialize the Engineered Model
    
    Attributes
    ----------
    curv_params : dict
        the parameters used for creating the gabor filters. The number of filters in this layer = n_ories x n_curves x number of frequencies
    
    filters_2 : str
        number of random filters used in conv layer 2
    
    batches_2 : str 
        the number of batches used to apply conv layer 2 filters. Can be used for larger number of filters to avoid memory issues 
    """
    
    def __init__(self, curv_params = {'n_ories':8,'n_curves':3,'gau_sizes':(5,),'spatial_fre':[1.2]},
                 filters_2=2000,filters_3=10000,batches_3 = 1):
    
        
        self.curv_params = curv_params
        self.filters_1 = self.curv_params['n_ories']*self.curv_params['n_curves']*len(self.curv_params['gau_sizes']*len(self.curv_params['spatial_fre']))
        self.filters_2 = filters_2
        self.filters_3 = filters_3
        self.batches_3 = batches_3
    
    
    
    def Build(self):
    
        c1 = StandardConvolution(filter_size=11,filter_type='curvature',curv_params=self.curv_params)     
        gb1 = GaussianBlur(kernel_size = 11)
        mp1 = nn.MaxPool2d(kernel_size=3)
        
        c2 = nn.Conv2d(24, self.filters_2, kernel_size=(9, 9))
        gb2 = GaussianBlur(kernel_size = 7)
        mp2 = nn.MaxPool2d(kernel_size=2)
        
        c3 = nn.Conv2d(self.filters_2, self.filters_3, kernel_size=(7,7))
        gb3 = GaussianBlur(kernel_size = 5)
        mp3 = nn.MaxPool2d(kernel_size=2)

        nl1 = nonlinearity('abs')
        
        last = Output()

        return Model(c1,gb1,mp1,c2,gb2,mp2,c3,gb3,mp3,self.batches_3,nl1,last)  
    