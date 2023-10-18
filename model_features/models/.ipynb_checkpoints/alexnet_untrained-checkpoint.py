import sys
import torchvision
import torch
from torch import nn
import pickle
import os
from model_features.layer_operations.output import Output

torch.manual_seed(0)
torch.cuda.manual_seed(0)
model = torchvision.models.alexnet(pretrained=False)


class Model(nn.Module):
    
    
    def __init__(self,
                features_layer:int,
                gpool: bool,
                last:nn.Module,
                ):
        
        super(Model, self).__init__()
        

        self.features_layer = features_layer
        self.gpool = gpool
        self.last = last
        
        
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
                    
        if self.gpool:
            H = x.shape[-1]
            gmp = nn.MaxPool2d(H)
            x = gmp(x)
            
        
        x = self.last(x)

        
        return x    


    
    
    
class AlexnetU:

    
    def __init__(self, features_layer:int = 12, 
                 gpool:int = True):
    
        self.features_layer = features_layer
        self.gpool = gpool
    
    def Build(self):

        last = Output()
        
        return Model(    
                features_layer = self.features_layer,
                gpool = self.gpool,
                last = last)