import sys
import torchvision
from models.layer_operations.output import Output
from models.layer_operations.pca import SpatialPCA, NormalPCA
import torch
from torch import nn
import pickle
import os
from models.layer_operations.random_proj import RandomProjection
from models.layer_operations.output import Output

torch.manual_seed(0)
torch.cuda.manual_seed(0)
model = torchvision.models.alexnet(pretrained=False)


def load_pca_file(identifier):

    file = open(os.path.join(PATH_TO_PCA,identifier), 'rb')
    _pca = pickle.load(file)  
    file.close()
    return _pca


class Model(nn.Module):
    
    
    def __init__(self,
                features_layer:int,
                global_mp: bool,
                rp: nn.Module,
                last:nn.Module,
                print_shape:bool=True
                ):
        
        super(Model, self).__init__()
        

        self.features_layer = features_layer
        self.global_mp = global_mp
        self.rp = rp
        self.last = last
        self.print_shape = print_shape
        
        
    def forward(self, x:nn.Module):
                
        
        # extract activations from L4
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach().cuda()
            return hook

        model.features[self.features_layer].register_forward_hook(get_activation(f'features.{self.features_layer}'))
        model.to('cuda')
        output = model(x.cuda())
        
        x = activation[f'features.{self.features_layer}']   
                    
        if self.global_mp:
            H = x.shape[-1]
            gmp = nn.MaxPool2d(H)
            x = gmp(x)
            print('gmp', x.shape)
            
        
        if self.rp is not None:
            x = self.rp(x)
            print('rp', x.shape)    
        
        x = self.last(x)
        if self.print_shape:
            print('output', x.shape)    
        
        return x    


    
    
    
class AlexnetU:

    
    def __init__(self, features_layer:int = 12, global_mp:int = False, num_projections:int = None):
    
        self.features_layer = features_layer
        self.num_projections = num_projections
        self.global_mp = global_mp
    
    def Build(self):
        
        rp = None
        if self.num_projections is not None:
            rp = RandomProjection(out_channels=self.num_projections)

        last = Output()
        
        return Model(    
                features_layer = self.features_layer,
                global_mp = self.global_mp,
                rp =rp,
                last = last)