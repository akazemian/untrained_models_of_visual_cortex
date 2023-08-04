import sys
import torchvision
import torch
from torch import nn
import pickle
import os
from model_features.layer_operations.output import Output

model = torchvision.models.alexnet(pretrained=True)



class Model(nn.Module):
    
    
    def __init__(self,
                features_layer: str,
                global_mp: bool,
                last:nn.Module,
                ):
        
        super(Model, self).__init__()
        

        self.features_layer = features_layer
        self.global_mp = global_mp
        self.last = last
        
        
    def forward(self, x):
                
        
        # extract activations from 
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

        x = self.last(x)
        
        return x    


    
    
    
class Alexnet:

    
    def __init__(self, features_layer:str = 12, global_mp:int = True):
    
        self.features_layer = features_layer
        self.global_mp = global_mp
    
    def Build(self):
    
        last = Output()
        
        return Model(    
                features_layer = self.features_layer,
                global_mp = self.global_mp,
                last = last)