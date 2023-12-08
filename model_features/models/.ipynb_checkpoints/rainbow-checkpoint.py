from model_features.layer_operations.output import Output
from model_features.models.learned_scaterring.main_custom import load_model
import torch
from torch import nn


rainbow_model = load_model().cuda()


class Model(nn.Module):
    
    def __init__(self,
                last: nn.Module,
                device: str,
                global_pool: bool,
                ):
        
        super(Model, self).__init__()
        
        self.global_pool = global_pool
        self.last = last
        self.device = device
        
    def forward(self, x:nn.Module):
                
        x = x.to(self.device)
        x = rainbow_model(x)
            
        print(x.shape)
                            
        mp = nn.MaxPool2d(x.shape[-1]//3)
        x = mp(x)
        print(x.shape)
        
        x = self.last(x)
        print(x.shape)
        return torch.Tensor(x).cuda()    


  

    
class RainbowModel():
        
    def __init__(self, 
                 global_pool:bool = False, 
                 device:str = 'cuda'):
    

        self.global_pool = global_pool
        self.device = device
    
    def Build(self):

        last = Output()

        return Model(
                global_pool = self.global_pool,
                last = last,
                device= self.device
        )


