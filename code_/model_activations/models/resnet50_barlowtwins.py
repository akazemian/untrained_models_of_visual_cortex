import torchvision
import torch
from torch import nn

from code_.model_activations.models.layer_operations.output import Output

torch.manual_seed(0)
torch.cuda.manual_seed(0)
# Load the pretrained Barlow Twins ResNet-50 model
resnet50 = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
resnet50.eval()  # Set the model to evaluation mode

class Model(nn.Module):
    """Pretrained alexnet with layer extraction"""
    def __init__(self, features_layer: int, last:nn.Module, device: str) -> None:
        super(Model, self).__init__()
        self.features_layer = features_layer
        self.last = last
        self.device = device
        self.model = resnet50.to(self.device)
        
    def forward(self, x) -> torch.Tensor:
        x = x.to(self.device)
        """Forward pass through the network, extracting activations from specific layer"""
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach().to(self.device)
            return hook
        
        match self.features_layer:
            case 1:
                self.model.layer1.register_forward_hook(get_activation(f"layer{self.features_layer}"))
            case 2:
                self.model.layer2.register_forward_hook(get_activation(f"layer{self.features_layer}"))
            case 3:
                self.model.layer3.register_forward_hook(get_activation(f"layer{self.features_layer}"))
            case 4:
                self.model.layer4.register_forward_hook(get_activation(f"layer{self.features_layer}"))  
        
        output = self.model(x).to(self.device)
        x = activation[f'layer{self.features_layer}']  
        if self.features_layer == 1:
            pool = nn.AvgPool2d(kernel_size=3)
            x = pool(x)
            
        elif self.features_layer in [2,3]:
            pool = nn.AvgPool2d(kernel_size=2)
            x = pool(x)
        return self.last(x)   

class ResNet50:
    """Constructing alexnet with the same structure as the expansion model"""
    def __init__(self, features_layer:int=4, device: str='cuda') -> None:
        self.features_layer = features_layer
        self.device = device
    def build(self):
        """Builds the model folliwng the expansion model configuration"""
        last = Output()
        return Model(self.features_layer, last, self.device)