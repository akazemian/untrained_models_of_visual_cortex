from layer_operations.sparse_random_proj import SparseRandomProjection
from layer_operations.output import Output
from layer_operations.random_proj import RandomProjection
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
                last: nn.Module,
                device: str,
                random_proj:int, 
                max_pool:bool, 
                global_pool: bool,
                ):
        
        super(Model, self).__init__()
        
        self.J = J
        self.L = L
        self.M = M
        self.N = N
        self.random_proj = random_proj
        self.max_pool = max_pool
        self.global_pool = global_pool
        self.last = last
        
        self.device = device
        
    def forward(self, x:nn.Module):
                
        x = x.to(self.device)
        N = x.shape[0]
        S = Scattering2D(J = self.J, shape=(self.M, self.N), L=self.L).cuda()
        x = S.scattering(x).squeeze()
            
        print(x.shape)
        
        if self.max_pool:
            mp = nn.MaxPool2d(x.shape[-1]//8)
            x = mp(x.flatten(start_dim=1,end_dim=2))
                    
        if self.global_pool:
            gmp = nn.MaxPool2d(x.shape[-1])
            x = gmp(x.flatten(start_dim=1,end_dim=2))
            
        if self.random_proj is not None:
            x = x.reshape(N,-1)
            print(x.shape)
            rp = SparseRandomProjection(n_components = self.random_proj)
            x = rp(x.to('cuda'))
            
            
        
        #x = x.reshape(N,-1)
        x = self.last(x)
        print(x.shape)
        return torch.Tensor(x).cuda()    


  

    
class ScatTransformKymatio():
        
    def __init__(self, J:int, 
                 L:int=8, 
                 M:int=224, 
                 N:int=224, 
                 random_proj:int = None,
                 max_pool:bool = True, 
                 global_pool:bool = False, 
                 device:str = 'cuda'):
    
        self.J = J
        self.L = L
        self.M = M
        self.N = N
        self.random_proj = random_proj
        self.max_pool = max_pool
        self.global_pool = global_pool
        self.device = device
    
    def Build(self):

        last = Output()

        return Model(
                J = self.J,
                L = self.L,
                M = self.M,
                N = self.N,
                random_proj = self.random_proj,
                max_pool = self.max_pool,
                global_pool = self.global_pool,
                last = last,
                device= self.device
        )


