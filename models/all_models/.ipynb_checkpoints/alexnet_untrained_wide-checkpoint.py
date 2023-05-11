import sys
import torchvision
from models.layer_operations.output import Output
import torch
from torch import nn
torch.manual_seed(0)
torch.cuda.manual_seed(0)
from models.layer_operations.output import Output
from models.layer_operations.random_proj import RandomProjection

model = torchvision.models.alexnet(pretrained=False)




class Model(nn.Module):
    
    
    def __init__(self,
                c5: nn.Module,
                r5: nn.Module,
                mp5: nn.Module,
                global_mp: bool,
                rp: nn.Module,
                 last: nn.Module,
                batches_5: int,
                print_shape: bool = True
                ):
        
        super(Model, self).__init__()
        
        self.c5 = c5
        self.r5 = r5
        self.mp5 = mp5
        self.global_mp = global_mp
        self.rp = rp
        self.last = last
        self.batches_5 = batches_5
        self.print_shape = print_shape
        
        
    def forward(self, x:nn.Module):
                
        
        
        # extract activations from L4
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach().cuda()
            return hook

        model.features[9].register_forward_hook(get_activation('features.9'))
        model.to('cuda')
        output = model(x.cuda())
        
        x = activation['features.9']
        if self.print_shape:
            print('layer 4', x.shape)        
        
        
        #conv layer 2
        conv_5 = []
        for i in range(self.batches_5):
            conv_5.append(self.c5(x.cuda())) 
        x = torch.cat(conv_5,dim=1)
        if self.print_shape:
            print('conv5', x.shape)
            
        
        x = self.r5(x)
        if self.print_shape:
            print('relu5', x.shape)        
        
        
        x = self.mp5(x)    
        if self.print_shape:
            print('maxpool5', x.shape)
        
        
        if self.global_mp:
            H = x.shape[-1]
            gmp = nn.MaxPool2d(H)
            x = gmp(x)
            print('gmp', x.shape)
            
        
        if self.rp is not None:
            x = self.rp(x)
            print('rp', x.shape)
        
        x = self.last(x)
        if self.print_shape:
            print('output', x.shape)
        
        return x    


    
    
    
    
    
    
    
class AlexnetUWide:

    
    def __init__(self, filters_5:int = 10000, batches_5:int = 1, global_mp:bool = False, num_projections:int = None):
    
        self.filters_5 = filters_5 
        self.batches_5 = batches_5
        self.num_projections = num_projections
        self.global_mp = global_mp

    
    def Build(self):
        
        c5 = nn.Conv2d(256, self.filters_5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        r5 = nn.ReLU()
        mp5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        rp = None
        if self.num_projections is not None:
            rp = RandomProjection(out_channels=self.num_projections)
        last = Output()

        return Model(c5=c5,
                     r5=r5,
                     mp5 = mp5,
                     global_mp = self.global_mp,
                     rp = rp,
                     last =last,
                     batches_5=self.batches_5)  
