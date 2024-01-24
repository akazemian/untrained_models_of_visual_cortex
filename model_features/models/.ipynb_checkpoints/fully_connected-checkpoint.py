from model_features.layer_operations.convolution import Convolution
from model_features.layer_operations.output import Output
from model_features.layer_operations.blurpool import BlurPool
from model_features.layer_operations.nonlinearity import NonLinearity
import torch
from torch import nn
                         
class Model3L(nn.Module):
    
    
    def __init__(self,
                lin1: nn.Module,
                 lin2: nn.Module,
                 lin3: nn.Module,
                 nl: nn.Module,
                 last: nn.Module,
                ):
        
        super(Model3L, self).__init__()
        
        self.lin1 = lin1
        self.lin2 = lin2
        self.lin3 = lin3

        
        self.nl = nl
        self.last = last
        
        
    def forward(self, x:nn.Module): 
       
        N = x.shape[0]
        x = self.lin1(x.reshape(N,-1))  # linear layer
        x = self.nl(x)
        print(x.shape)
        
        x = self.lin2(x)
        x = self.nl(x)
        print(x.shape)
        
        x = self.lin3(x)
        x = self.nl(x)
        print(x.shape)
        
        x = self.last(x)
    
        return x    


    
class FullyConnected3L:


    def __init__(self, 
                 image_size:int = 224,
                 features_1:int = 108,
                 features_2:int = 1000,
                 features_3:int = 3000):    
        
        self.input_dim = 224
        self.features_1 = features_1
        self.features_2 = features_2
        self.features_3 = features_3*9*9
        self.input_dim = image_size**2*3
    
    
    def Build(self):        
        
        lin1 = nn.Linear(self.input_dim, self.features_1)
        lin2 = nn.Linear(self.features_1, self.features_2)
        lin3 = nn.Linear(self.features_2, self.features_3)
        
        nl = NonLinearity('relu')
        last = Output()
        
        return Model3L(lin1, lin2, lin3, nl, last)
    
    
    
    
    
    


class Model5L(nn.Module):
    
    
    def __init__(self,
                lin1: nn.Module,
                 lin2: nn.Module,
                 lin3: nn.Module,
                 lin4: nn.Module,
                 lin5: nn.Module,
                 nl: nn.Module,
                 last: nn.Module,
                ):
        
        super(Model5L, self).__init__()
        
        self.lin1 = lin1
        self.lin2 = lin2
        self.lin3 = lin3
        self.lin4 = lin4
        self.lin5 = lin5
        
        self.nl = nl
        self.last = last
        
        
    def forward(self, x:nn.Module): 
       
        N = x.shape[0]
        x = self.lin1(x.reshape(N,-1))  # linear layer
        x = self.nl(x)
        print(x.shape)
        
        x = self.lin2(x)
        x = self.nl(x)
        print(x.shape)
        
        x = self.lin3(x)
        x = self.nl(x)
        print(x.shape)
        
        x = self.lin4(x)
        x = self.nl(x)
        print(x.shape)
        
        x = self.lin5(x)
        x = self.nl(x)
        print(x.shape)
        
        x = self.last(x)
    
        return x    


    
  

    
class FullyConnected5L:


    def __init__(self, 
                 image_size:int = 224,
                 features_1:int = 108,
                 features_2:int = 1000,
                 features_3:int = 5000,
                features_4:int = 5000,
                features_5:int = 3000):    
        
        self.features_1 = features_1
        self.features_2 = features_2
        self.features_3 = features_3
        self.features_4 = features_4
        self.features_5 = features_5*6*6
        self.input_dim = image_size**2*3
    
    
    def Build(self):        
        
        lin1 = nn.Linear(self.input_dim, self.features_1)
        lin2 = nn.Linear(self.features_1, self.features_2)
        lin3 = nn.Linear(self.features_2, self.features_3)
        lin4 = nn.Linear(self.features_3, self.features_4)
        lin5 = nn.Linear(self.features_4, self.features_5)
        
        
        nl = NonLinearity('relu')
        last = Output()
        
        return Model5L(lin1, lin2, lin3, lin4, lin5, nl, last)
    