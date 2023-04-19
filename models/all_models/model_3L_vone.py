from models.engineered_model import Model
from models.layer_operations.convolution import StandardConvolution,RandomProjections
from models.layer_operations.output import Output
from models.layer_operations.voneblock import *

from models.layer_operations.convolution import *
from models.layer_operations.output import Output
import torch
from torch import nn
                         


class Model(nn.Module):
    
    
    def __init__(self,
                vone: nn.Module,
                mp1: nn.Module,
                c2: nn.Module,
                mp2: nn.Module,
                c3: nn.Module,
                mp3: nn.Module,
                batches_3: int,
                last: nn.Module,
                print_shape: bool = True
                ):
        
        super(Model, self).__init__()
        
        
        self.vone = vone 
        self.mp1 = mp1
        self.c2 = c2
        self.mp2 = mp2
        self.c3 = c3
        self.mp3 = mp3
        self.batches_3 = batches_3
        self.last = last
        self.print_shape = print_shape
        
        
    def forward(self, x:nn.Module):
                
        
        #conv layer 1
        x = self.vone(x)
        if self.print_shape:
            print('vone', x.shape)
    
        x = self.mp1(x)
        if self.print_shape:
            print('mp1', x.shape)
            
        #conv layer 2
        x = self.c2(x)
        if self.print_shape:
            print('conv2', x.shape)
            
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
            
        x = self.mp3(x)
        if self.print_shape:
            print('mp3', x.shape)
        
        x = self.last(x)
        if self.print_shape:
            print('output', x.shape)
        
        return x    



  
class EngineeredModel3LVOne:
    
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
    
    def __init__(self, filters_2=2000,filters_3=10000,batches_3 = 1,im_size=224):
    
        
        self.filters_2 = filters_2
        self.filters_3 = filters_3
        self.batches_3 = batches_3
        self.im_size = im_size
    
    
    
    def Build(self):
    
        sf_corr=0.75
        sf_max=9
        sf_min=0
        rand_param=False
        gabor_seed=0
        simple_channels=256
        complex_channels=256
        noise_mode='neuronal'
        noise_scale=0.35
        noise_level=0.07
        k_exc=25
        image_size=self.im_size
        visual_degrees=8
        ksize=25
        stride=4


        out_channels = simple_channels + complex_channels
        sf, theta, phase, nx, ny = generate_gabor_param(out_channels, gabor_seed, rand_param, sf_corr, sf_max, sf_min)
        gabor_params = {'simple_channels': simple_channels, 'complex_channels': complex_channels, 'rand_param': rand_param,
                        'gabor_seed': gabor_seed, 'sf_max': sf_max, 'sf_corr': sf_corr, 'sf': sf.copy(),
                        'theta': theta.copy(), 'phase': phase.copy(), 'nx': nx.copy(), 'ny': ny.copy()}
        # Conversions
        ppd = image_size / visual_degrees
        sf = sf / ppd
        sigx = nx / sf
        sigy = ny / sf
        theta = theta/180 * np.pi
        phase = phase / 180 * np.pi

        vone = VOneBlock(sf=sf, theta=theta, sigx=sigx, sigy=sigy, phase=phase,
                               k_exc=k_exc, noise_mode=noise_mode, noise_scale=noise_scale, noise_level=noise_level,
                               simple_channels=simple_channels, complex_channels=complex_channels,
                               ksize=ksize, stride=stride, input_size=image_size)

        vone.image_size = image_size
        vone.visual_degrees = visual_degrees
        vone.gabor_params = gabor_params
    
        mp1 = nn.MaxPool2d(kernel_size=2)
        c2 = nn.Conv2d(out_channels, self.filters_2, kernel_size=(9, 9))
        mp2 = nn.MaxPool2d(kernel_size=2)
        c3 = nn.Conv2d(self.filters_2, self.filters_3, kernel_size=(3,3))
        mp3 = nn.MaxPool2d(kernel_size=2)

        last = Output()

        return Model(vone,mp1,c2,mp2,c3,mp3,self.batches_3,last)  
    
    
