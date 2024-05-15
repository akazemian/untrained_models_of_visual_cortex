import sys
import torchvision
import torch
from torch import nn
from layer_operations.output import Output

torch.manual_seed(0)
torch.cuda.manual_seed(0)
alexnet_trained = torchvision.models.alexnet(pretrained=True)

class Model(nn.Module):
    """Pretrained alexnet with layer extraction"""
    def __init__(self, features_layer: int, last:nn.Module) -> None:
        super(Model, self).__init__()
        self.features_layer = features_layer
        self.last = last
        self.device =  device
        
    def forward(self, x) -> torch.Tensor:
        """Forward pass through the network, extracting activations from specific layer"""
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach().cuda()
            return hook

        model.features[self.features_layer].register_forward_hook(get_activation(f'features.{self.features_layer}'))
        model.to(self.device)
        output = model(x.to(self.device))
        x = activation[f'features.{self.features_layer}']           
        return self.last(x)   

class Alexnet:
    """Constructing alexnet with the same structure as the expansion model"""
    def __init__(self, features_layer:str = 12, , device: str = 'cuda') -> None:
        self.features_layer = features_layer
    def build(self):
        """Builds the model folliwng the expansion model configuration"""
        last = Output()
        return Model(self.features_layer, last, self.device)