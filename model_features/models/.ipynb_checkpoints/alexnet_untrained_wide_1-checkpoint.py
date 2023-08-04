import sys
import torchvision
from models.layer_operations.output import Output
import torch
from torch import nn
torch.manual_seed(0)
torch.cuda.manual_seed(0)
model = torchvision.models.alexnet(pretrained=False)




class Model(nn.Module):
    
    
    def __init__(self,
                c5: nn.Module,
                r5: nn.Module,
                mp5: nn.Module,
                last: nn.Module,
                batches_5: int,
                print_shape: bool = True
                ):
        
        super(Model, self).__init__()
        
        self.c5 = c5
        self.r5 = r5
        self.mp5 = mp5
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
        
        
        x = self.last(x)
        if self.print_shape:
            print('output', x.shape)
        
        return x    


    
    
    
    
    
    
    
class AlexnetU1:

    
    def __init__(self, filters_5 = 10000, batches_5=1):
    
        self.filters_5 = filters_5 
        self.batches_5 = batches_5

    
    def Build(self):
        
        c5 = nn.Conv2d(256, self.filters_5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        r5 = nn.ReLU()
        mp5 = nn.MaxPool2d(kernel_size=4)
        last = Output()

        return Model(c5,r5,mp5,last,self.batches_5)  
