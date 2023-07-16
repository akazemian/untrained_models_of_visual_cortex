from torch import nn

class Output(nn.Module):
    
    def __init__(self):
        super().__init__()
                
    def forward(self,x):
        N = x.shape[0]
        return x.reshape(N,-1)