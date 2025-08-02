import timm
import torch
from torch import nn
from code_.model_activations.models.layer_operations.output import Output
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Load pre-trained ViT model
model_name = "vit_base_patch16_224"
vit_trained = timm.create_model(model_name, pretrained=False, num_classes=1000)
vit_trained.eval()

class Model(nn.Module):
    """Pretrained alexnet with layer extraction"""
    def __init__(self, block: int, last:nn.Module, device: str) -> None:
        super(Model, self).__init__()
        self.block = block
        self.last = last
        self.device = device
        self.model = vit_trained.to(self.device)
        
    def forward(self, x) -> torch.Tensor:
        activations = {}  # Global dict to store the activation
        x = x.to(self.device)
        """Forward pass through the network, extracting activations from specific layer"""
        activation = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output
            return hook

        # Register forward hook on Block #5 (indexing starts at 0)
        self.model.blocks[self.block].register_forward_hook(get_activation(f"blocks[{self.block}]"))
        output = self.model(x).to(self.device)
        x = activation[f"blocks[{self.block}]"]           
        print('test shape', self.last(x).shape)
        return self.last(x)   

class ViTU:
    """Constructing alexnet with the same structure as the expansion model"""
    def __init__(self, block:int = 11, device: str='cuda') -> None:
        self.block = block
        self.device = device
    def build(self):
        """Builds the model folliwng the expansion model configuration"""
        last = Output()
        return Model(self.block, last, self.device)