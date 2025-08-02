import torch
from torch import nn
from code_.model_activations.models.layer_operations.output import Output
from code_.model_activations.models.layer_operations.nonlinearity import NonLinearity

# Set random seeds globally (if needed elsewhere, set it again in that context)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

class Model5L(nn.Module):
    """MLP architecture consisting of 5 layers."""
    def __init__(self, lin1: nn.Module, lin2: nn.Module, lin3: nn.Module, lin4: nn.Module,
                 lin5: nn.Module, nl: nn.Module, last: nn.Module, device: str) -> None:
        super(Model5L, self).__init__()
        self.lin1 = lin1
        self.lin2 = lin2
        self.lin3 = lin3
        self.lin4 = lin4
        self.lin5 = lin5
        self.nl = nl
        self.last = last
        self.device = device        
        
        
    def forward(self, x:nn.Module):         
   
        print('reading old model')
        
        x = x.to(self.device)
        x = x.flatten(1, -1)
        
        #layer 1 
        x = self.lin1(x)  # conv 
        x = self.nl(x) # non linearity 
        
        #layer 2
        x = self.lin2(x)  
        x = self.nl(x) 
            
        #layer 3
        x = self.lin3(x)  
        x = self.nl(x) 
        
        #layer 4
        x = self.lin4(x)  
        x = self.nl(x) 
        
        #layer 5
        x = self.lin5(x) 
        x = self.nl(x) 
        
        x = self.last(x) # final layer
        
        return x    

class FullyConnected5L:
    """Constructing the 5-layer MLP."""
    def __init__(self, image_size:int = 224, features_1:int = 108, features_2:int = 1000, 
                 features_3:int = 3000, features_4:int = 5000, features_5:int = 30000, device:str='cuda') -> None:
        self.features_1 = features_1
        self.features_2 = features_2
        self.features_3 = features_3
        self.features_4 = features_4
        self.features_5 = features_5 * 6**2
        self.input_dim = 3 * image_size**2
        self.device = device
    
    def build(self) -> Model5L:        
        """Builds the complete model using specified layer sizes."""
        lin1 = nn.Linear(self.input_dim, self.features_1)
        lin2 = nn.Linear(self.features_1, self.features_2)
        lin3 = nn.Linear(self.features_2, self.features_3)
        lin4 = nn.Linear(self.features_3, self.features_4)
        lin5 = nn.Linear(self.features_4, self.features_5)
        nl = NonLinearity('relu')
        last = Output()
        
        return Model5L(lin1, lin2, lin3, lin4, lin5, nl, last,self.device)

# import torch
# from torch import nn
# from model_features.models.layer_operations.output import Output
# from model_features.models.layer_operations.nonlinearity import NonLinearity

# # Set random seeds globally (if needed elsewhere, set it again in that context)
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

# class Model5L(nn.Module):
#     """MLP architecture consisting of 5 layers."""
#     def __init__(self, lin1: nn.Module, lin2: nn.Module, lin3: nn.Module, lin4: nn.Module,
#                  lin5: nn.Module, nl: nn.Module, last: nn.Module) -> None:
#         super(Model5L, self).__init__()
#         self.lin1 = lin1
#         self.lin2 = lin2
#         self.lin3 = lin3
#         self.lin4 = lin4
#         self.lin5 = lin5
#         self.nl = nl
#         self.last = last
        
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass through the network, applying each layer sequentially."""
#         x = x.flatten(1, -1)
#         # Processing layers 1 to 5
#         for i in range(1, 6):
#             lin = getattr(self, f'lin{i}')
#             x = lin(x)
#             x = self.nl(x)
#         # Output layer
#         x = self.last(x)
#         return x

# class FullyConnected5L:
#     """Constructing the 5-layer MLP."""
#     def __init__(self, image_size:int = 224, features_1:int = 108, features_2:int = 1000, 
#                  features_3:int = 3000, features_4:int = 5000, features_5:int = 3000) -> None:
#         self.features_1 = features_1
#         self.features_2 = features_2
#         self.features_3 = features_3
#         self.features_4 = features_4
#         self.features_5 = features_5 * 6**2
#         self.input_dim = 3 * image_size**2
    
#     def build(self) -> Model5L:        
#         """Builds the complete model using specified layer sizes."""
#         lin1 = nn.Linear(self.input_dim, self.features_1)
#         lin2 = nn.Linear(self.features_1, self.features_2)
#         lin3 = nn.Linear(self.features_2, self.features_3)
#         lin4 = nn.Linear(self.features_3, self.features_4)
#         lin5 = nn.Linear(self.features_4, self.features_5)
#         nl = NonLinearity('relu')
#         last = Output()
        
#         return Model5L(lin1, lin2, lin3, lin4, lin5, nl, last,)
