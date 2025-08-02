import sys
import torchvision
from models.layer_operations.output import Output
import torch
from torch import nn
torch.manual_seed(0)
torch.cuda.manual_seed(0)
from torch.nn import Conv2d, ReLU, MaxPool2d





class Model(nn.Module):
    
    
    def __init__(self,
                conv1 : nn.Module,
                relu1 : nn.Module,
                mp1 : nn.Module,
                conv2 : nn.Module,
                relu2 : nn.Module,
                mp2 : nn.Module,
                conv3 : nn.Module,
                relu3 : nn.Module,
                conv4 : nn.Module,
                relu4 : nn.Module,
                mp4: nn.Module,
                conv5 : nn.Module,
                relu5 : nn.Module,
                mp5: nn.Module,
                last: nn.Module,
                print_shape: bool = True
                ):
        
        super(Model, self).__init__()
        

        self.conv1 = conv1 
        self.relu1 = relu1 
        self.mp1 = mp1 
        
        self.conv2 = conv2 
        self.relu2 = relu2 
        self.mp2 = mp2 
        
        self.conv3 = conv3 
        self.relu3 = relu3 
        
        self.conv4 = conv4 
        self.relu4 = relu4 
        self.mp4 = mp4
        
        self.conv5 = conv5 
        self.relu5 = relu5 
        self.mp5 = mp5 
        
        self.last = last
        self.print_shape = print_shape
        
    def forward(self, x:nn.Module):
                

        x = self.conv1(x) 
        x = self.relu1(x)
        if self.print_shape:
            print('conv1', x.shape)
        x = self.mp1 (x)
        if self.print_shape:
            print('mp1', x.shape)
            
            
            
        x = self.conv2(x)
        if self.print_shape:
            print('conv2', x.shape)
        x = self.relu2(x)
        x = self.mp2(x)
        if self.print_shape:
            print('mp2', x.shape)
            
            
            
        x = self.conv3(x)
        if self.print_shape:
            print('conv3', x.shape)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
        if self.print_shape:
            print('conv4', x.shape)
        # x = self.mp4(x)
        # if self.print_shape:
        #     print('mp4', x.shape)
            
            
        x = self.conv5(x)
        x = self.relu5(x)
        if self.print_shape:
            print('conv5', x.shape)
        x = self.mp5(x)
        if self.print_shape:
            print('mp5', x.shape)
            
            
        x = self.last(x)
        if self.print_shape:
            print('output', x.shape)
        
        return x    


    
    
    
    
    
    
    
class AlexnetU2:

    
    def __init__(self, filters_5 = 10000):
    
        self.filters_5 = filters_5        

    
    def Build(self):

        conv1 = Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        relu1 = ReLU(inplace=True)
        mp1 = MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        conv2 = Conv2d(64, 1000, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        relu2 = ReLU(inplace=True)
        mp2 = MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        conv3 = Conv2d(1000, 5000, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        relu3 = ReLU(inplace=True)
        
        conv4 = Conv2d(5000, 2000, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        relu4 = ReLU(inplace=True)
        mp4 = MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        conv5 = Conv2d(2000, self.filters_5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        relu5 = ReLU(inplace=True)
        mp5 = MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)

        last = Output()
        
        return Model(conv1, relu1, mp1,
                    conv2,relu2, mp2,
                    conv3, relu3,
                    conv4, relu4, mp4,
                    conv5, relu5, mp5,
                    last)  