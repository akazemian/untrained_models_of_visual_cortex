import sys
import torchvision
from models.layer_operations.output import Output
from models.layer_operations.pca import SpatialPCA, NormalPCA
import torch
from torch import nn
model = torchvision.models.alexnet(pretrained=True)
import pickle
import os

ROOT_DATA = os.getenv('MB_DATA_PATH')
PATH_TO_PCA = os.path.join(ROOT_DATA,'pca')
IDEN = 'alexnet_mp_pca_256_naturalscenes'


def load_pca_file(identifier):

    file = open(os.path.join(PATH_TO_PCA,identifier), 'rb')
    _pca = pickle.load(file)  
    file.close()
    return _pca


class Model(nn.Module):
    
    
    def __init__(self,
                pca5: nn.Module,
                global_mp:bool,
                print_shape:bool=True
                ):
        
        super(Model, self).__init__()
        

        self.global_mp = global_mp
        self.pca5 = pca5
        self.print_shape = print_shape
        
        
    def forward(self, x:nn.Module):
                
        
        # extract activations from L4
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach().cuda()
            return hook

        model.features[12].register_forward_hook(get_activation('features.12'))
        model.to('cuda')
        output = model(x.cuda())
        
        x = activation['features.12']   
                    
        if self.global_mp:
            H = x.shape[-1]
            gmp = nn.MaxPool2d(H)
            x = gmp(x)
            print('gmp', x.shape)
        
        x = self.pca5(x)    
        if self.print_shape:
            print('pca5', x.shape)       
        
        return x    


    
    
    
class AlexnetPCA:

    
    def __init__(self, n_components=5000, global_mp = False):
    
        self._pca5 = load_pca_file(IDEN)
        self.n_components = n_components
        self.global_mp = global_mp
    
    def Build(self):
        
        pca5 = NormalPCA(_pca = self._pca5, n_components = self.n_components)
        
        return Model(pca5,self.global_mp)  
