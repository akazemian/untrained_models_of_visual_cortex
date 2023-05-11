
from models.layer_operations.output import Output
from models.layer_operations.random_proj import RandomProjection
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
                flatten: bool,
                global_mp: bool,
                rp: nn.Module,
                last: nn.Module,
                print_shape: bool = True
                ):
        
        super(Model, self).__init__()
        
        
        self.J = J
        self.L = L
        self.M = M
        self.N = N
        
        
        self.rp = rp
        self.flatten = flatten 
        self.global_mp = global_mp
        self.last = last
        self.print_shape = print_shape
        
        
    def forward(self, x:nn.Module):
                
        N = x.shape[0]
        S = Scattering2D(J = self.J, shape=(self.M, self.N), L=self.L).cuda()

        x = S.scattering(x).squeeze()
        print('st', x.shape)
        
            
        if self.global_mp:
            H = x.shape[-1]
            gmp = nn.MaxPool2d(2)
            x = gmp(x)
            print('gmp', x.shape)
            
            
        if self.flatten:
            x = x.reshape(N,-1)
            
            
        if self.rp is not None:
            x = self.rp(x)
            print('rp', x.shape)
            
            
        x = self.last(x)
        if self.print_shape:
            print('output', x.shape)
        
        return x    


  

    
class ScatTransformKymatio():
        
        
    def __init__(self, J, L, M, N, global_mp = False, flatten = False, num_projections=None):
    
        
        self.J = J
        self.L = L
        self.M = M
        self.N = N
        self.flatten = flatten
        self.global_mp = global_mp
        self.num_projections = num_projections
    
    
    
    def Build(self):

        rp = None
        if self.num_projections is not None:
            rp = RandomProjection(out_channels=self.num_projections)
        last = Output()



        return Model(
                J = self.J,
                L = self.L,
                M = self.M,
                N = self.N,
                flatten = self.flatten,
                global_mp = self.global_mp,
                rp = rp,
                last = last,
        )


