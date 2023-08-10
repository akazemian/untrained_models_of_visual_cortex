
from model_features.layer_operations.output import Output
from model_features.layer_operations.random_proj import RandomProjection
import torch
from torch import nn
from kymatio.torch import Scattering2D
import torch








class Model(nn.Module):
    
    
    def __init__(self,
                J: int,
                L: int,
                M: int,
                N: int,
                global_mp: bool,
                last: nn.Module,
                ):
        
        super(Model, self).__init__()
        
        
        self.J = J
        self.L = L
        self.M = M
        self.N = N
        
        
        self.global_mp = global_mp
        self.last = last
        
        
    def forward(self, x:nn.Module):
                
        x = x.to('cuda')
        N = x.shape[0]
        S = Scattering2D(J = self.J, shape=(self.M, self.N), L=self.L).cuda()

        x = S.scattering(x).squeeze()
        
            
        if self.global_mp:
            H = x.shape[-1]
            gmp = nn.MaxPool2d(H)
            x = gmp(x.flatten(start_dim=1,end_dim=2))

            
            
        x = x.reshape(N,-1)
        
        x = self.last(x)
        
        return x    


  

    
class ScatTransformKymatio():
        
        
    def __init__(self, J, L, M, N, global_mp = False):
    
        
        self.J = J
        self.L = L
        self.M = M
        self.N = N
        self.global_mp = global_mp
    
    
    
    def Build(self):


        last = Output()

        return Model(
                J = self.J,
                L = self.L,
                M = self.M,
                N = self.N,
                global_mp = self.global_mp,
                last = last,
        )


