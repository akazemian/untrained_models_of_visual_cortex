from model_features.models.layer_operations.convolution import Convolution
from model_features.models.layer_operations.output import Output
from model_features.models.layer_operations.nonlinearity import NonLinearity
import torch
from torch import nn
                         

    


class Model5L(nn.Module):
    
    
    def __init__(self,
                conv: nn.Module,
                 lin1: nn.Module,
                 lin2: nn.Module,
                 lin3: nn.Module,
                 lin4: nn.Module,
                 lin5: nn.Module,
                 nl: nn.Module,
                 last: nn.Module,
                ):
        
        super(Model5L, self).__init__()
        
        self.conv = conv
        self.lin1 = lin1
        self.lin2 = lin2
        self.lin3 = lin3
        self.lin4 = lin4
        self.lin5 = lin5
        
        self.nl = nl
        self.last = last
        
        
    def forward(self, x:nn.Module): 
       
        N = x.shape[0]
        
        # x = self.conv(x)
        # print(x.shape)

        x = self.lin1(x.reshape(N,-1))  # linear layer
        x = self.nl(x)
        
        x = self.lin2(x)
        x = self.nl(x)
        
        x = self.lin3(x)
        x = self.nl(x)
        
        x = self.lin4(x)
        x = self.nl(x)
        
        x = self.lin5(x)
        x = self.nl(x)
        
        x = self.last(x)
    
        return x    


    
  

    
class FullyConnected5L:


    def __init__(self, 
                 image_size:int = 224,
                 features_1:int = 108, #*112**2
                 features_2:int = 1000, #*53**2
                 features_3:int = 5000, #*25**2
                features_4:int = 5000, #*12**2
                features_5:int = 3000,device='cuda'): #*6**2
        
        self.features_1 = features_1
        self.features_2 = features_2
        self.features_3 = features_3
        self.features_4 = features_4
        self.features_5 = features_5*6**2
        self.device = device
        
        self.filter_params = {'type':'curvature','n_ories':12,'n_curves':3,'gau_sizes':(5,),'spatial_fre':[1.2]}
        self.input_dim = 3 * image_size**2
    
    
    def Build(self):        
        
        conv = Convolution(filter_size=15, filter_params=self.filter_params, device = self.device) 
        lin1 = nn.Linear(self.input_dim, self.features_1)
        lin2 = nn.Linear(self.features_1, self.features_2)
        lin3 = nn.Linear(self.features_2, self.features_3)
        lin4 = nn.Linear(self.features_3, self.features_4)
        lin5 = nn.Linear(self.features_4, self.features_5)
        
        
        nl = NonLinearity('relu')
        last = Output()
        
        return Model5L(conv, lin1, lin2, lin3, lin4, lin5, nl, last)
    