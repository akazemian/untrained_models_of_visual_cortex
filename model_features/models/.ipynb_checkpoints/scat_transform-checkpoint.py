from layer_operations.sparse_random_proj import SparseRandomProjection
from layer_operations.output import Output
from layer_operations.random_proj import RandomProjection
import torch
from torch import nn
from kymatio.torch import Scattering2D
import torch



class ScatTransform(nn.Module):
    
    def __init__(self,
                J: int,
                L: int,
                M: int,
                N: int):
        
        super(Model, self).__init__()
        
        self.J, self.L = L, self.M,self.N, J, L, M, N
        self.model = Scattering2D(J = self.J, shape=(self.M, self.N), L=self.L).cuda()        

    
    def forward(self, x:nn.Module):
        return self.model(x)
                
          


  

    
class ScatTransform():
        
    def __init__(self, J:int, 
                 L:int=8, 
                 M:int=64, 
                 N:int=64):
    
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


