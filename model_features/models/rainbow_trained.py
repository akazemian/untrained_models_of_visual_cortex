from model_features.layer_operations.output import Output
from model_features.models.learned_scaterring.main import load_model
import torch
from torch import nn




path = '/data/atlas/rainbow_models/Pr_Norm/batchsize_128_lrfreq_45_best.pth.tar'
#path = '/data/atlas/rainbow_models/Skip_Pc_Norm/batchsize_128_lrfreq_45_best.pth.tar'

rainbow_model = load_model().cuda()
checkpoint = torch.load(path)
state_dict = checkpoint["state_dict"]
state_dict = {key.replace("(0, 0)", "0"): value for key, value in state_dict.items()}
checkpoint["state_dict"] = state_dict
rainbow_model.load_state_dict(checkpoint['state_dict'])
rainbow_model = rainbow_model[:-6]

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
        
        try: 
            print(x.shape)
            
        except AttributeError:
            x = x.full_view()                   
            print(x.shape)
        
        
        mp = nn.MaxPool2d(x.shape[-1]//6)
        x = mp(x)      
        print(x.shape)
        
        
        x = self.last(x)
        print(x.shape)
        return torch.Tensor(x).cuda()    


  

    
class RainbowModelTrained():
        
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


