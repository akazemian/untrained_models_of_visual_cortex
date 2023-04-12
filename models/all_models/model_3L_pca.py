from models.engineered_model import Model
from models.layer_operations.convolution import StandardConvolution,RandomProjections
from models.layer_operations.output import Output
from models.layer_operations.pca import SpatialPCA
from models.layer_operations.convolution import *
from models.layer_operations.output import Output
import torch
from torch import nn
import os
import pickle


PATH_TO_PCA = '/data/atlas/pca'

# N_COMPONENTS_L2 = 200
# N_COMPONENTS_L3 = 500

N_COMPONENTS_L2 = 500
N_COMPONENTS_L3 = 1000

# IDEN_L2 = f'model_2L_mp_4000_nsd_pca_{N_COMPONENTS_L2}_components'
# IDEN_L3 = f'model_3L_mp_10000_nsd_pca_{N_COMPONENTS_L3}_components'
 
IDEN_L2 = f'model_2L_mp_5000_nsd_pca_{N_COMPONENTS_L2}_components'
IDEN_L3 = f'model_3L_mp_10000_nsd_pca_{N_COMPONENTS_L3}_components'

def load_pca_file(identifier):

    file = open(os.path.join(PATH_TO_PCA,identifier), 'rb')
    _pca = pickle.load(file)  
    file.close()
    return _pca
    
    

class Model(nn.Module):
    
    
    def __init__(self,
                c1: nn.Module,
                mp1: nn.Module,
                c2: nn.Module,
                mp2: nn.Module,
                pca2: nn.Module,
                c3: nn.Module,
                pca3: nn.Module,
                last: nn.Module,
                print_shape: bool = True
                ):
        
        super(Model, self).__init__()
        
        
        self.c1 = c1 
        self.mp1 = mp1
        self.c2 = c2
        self.mp2 = mp2
        self.pca2 = pca2
        self.c3 = c3
        self.pca3 = pca3
        self.last = last
        self.print_shape = print_shape
        
        
    def forward(self, x:nn.Module):
                
        
        #conv layer 1
        x = self.c1(x)
        if self.print_shape:
            print('conv1', x.shape)
    
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
            
            
        x = self.pca2(x)
        print('pca2', x.shape)
        
        #conv layer 3
        x = self.c3(x)
        if self.print_shape:
            print('conv3', x.shape)

            
        x = self.pca3(x)
        print('pca3', x.shape)            
        
        x = self.last(x)
        if self.print_shape:
            print('output', x.shape)
            
        return x    



  
class EngineeredModel3LPCA:
    
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
                 filters_2=5000,filters_3=10000):
    
        
        self.curv_params = curv_params
        self.filters_1 = self.curv_params['n_ories']*self.curv_params['n_curves']*len(self.curv_params['gau_sizes']*len(self.curv_params['spatial_fre']))
        self.filters_2 = filters_2
        self.filters_3 = filters_3

        self._pca2 = load_pca_file(IDEN_L2)
        self._pca3 = load_pca_file(IDEN_L3)
        
        
    def Build(self):
    
        c1 = StandardConvolution(filter_size=15,filter_type='curvature',curv_params=self.curv_params)     
        mp1 = nn.MaxPool2d(kernel_size=3)
        
        c2 = nn.Conv2d(24, self.filters_2, kernel_size=(9, 9))
        mp2 = nn.MaxPool2d(kernel_size=2)
        pca2 = SpatialPCA(_pca = self._pca2, n_components = N_COMPONENTS_L2)
        
        c3 = nn.Conv2d(N_COMPONENTS_L2, self.filters_3, kernel_size=(7,7))
        pca3 = SpatialPCA(_pca = self._pca3, n_components = N_COMPONENTS_L3)
        
        last = Output()


        return Model(
                c1 = c1,
                mp1 = mp1,
                c2 = c2,
                mp2 = mp2,
                pca2 = pca2,
                c3 = c3,
                pca3 = pca3,
                last = last,
                )