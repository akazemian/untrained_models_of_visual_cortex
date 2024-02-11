from model_features.layer_operations.preset_filters import filters
from torch.nn import functional as F
import math
import torch
torch.manual_seed(42)
from torch import nn
import numpy as np
import torch.nn.functional as F



class Convolution(nn.Module):

    
    def __init__(self, 
                 device:str,
                 filter_params:dict=None,
                 filter_size:int=None,
                ):
                
        super().__init__()
        

        self.filter_size = filter_size
        self.filter_params = filter_params
        self.device = device
    
    def extra_repr(self) -> str:
        return 'filter_size={filter_size}, filter_params:{filter_params}'.format(**self.__dict__)
    
    
    def forward(self,x):
            
        in_channels = x.shape[1]
        weights = filters(in_channels=1, kernel_size=self.filter_size, filter_params=self.filter_params).to(self.device)
        
        
        # for RGB input (the preset L1 filters are repeated across the 3 channels)
            
        convolved_tensor = []
        for i in range(in_channels):
            channel_image = x[:, i:i+1, :, :]
            channel_convolved = F.conv2d(channel_image, weight= weights, padding=math.floor(weights.shape[-1] / 2)).to(self.device)
            convolved_tensor.append(channel_convolved)
        x = torch.cat(convolved_tensor, dim=1)    
    

        return x
    




class NonSharedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(NonSharedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Ensure kernel_size and padding are treated as integers
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.stride = stride
        self.padding = padding if isinstance(padding, int) else padding[0]

        # Adjusted weight shape considering kernel_size as an int
        self.weight_shape = (out_channels, in_channels, self.kernel_size, self.kernel_size)


    def forward(self, x):
        # Get the input dimensions
        batch_size, _, height, width = x.shape
        
        # Calculate output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Output tensor
        output = torch.zeros((batch_size, self.out_channels, out_height, out_width), device=x.device, dtype=x.dtype)

        # Initialize random weights for each patch
        # This is highly inefficient and not practical but follows the request
        for b in range(batch_size):
            for i in range(out_height):
                for j in range(out_width):
                    # Create an uninitialized tensor for weights
                    weights = torch.empty(self.weight_shape, device=x.device, dtype=x.dtype)
                    # Apply Kaiming/He uniform initialization to the weights
                    nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
                    
                    h_start = i * self.stride
                    h_end = h_start + self.kernel_size
                    w_start = j * self.stride
                    w_end = w_start + self.kernel_size
                    
                    # Extract the patch
                    patch = x[b:b+1, :, h_start:h_end, w_start:w_end].unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
                    patch = patch.contiguous().view(self.in_channels, -1)
                    
                    # Convolve the patch with the weights
                    for c in range(self.out_channels):
                        weight = weights[c].view(-1)
                        output[b, c, i, j] = torch.dot(patch.view(-1), weight)
        
        return output


        
def initialize_conv_layer(conv_layer, initialization):
    
    match initialization:
        
        case 'kaiming_uniform':
            torch.nn.init.kaiming_uniform_(conv_layer.weight)
            
        case 'kaiming_normal':
            torch.nn.init.kaiming_normal_(conv_layer.weight)
            
        case 'orthogonal':
            torch.nn.init.orthogonal_(conv_layer.weight) 
            
        case 'xavier_uniform':
            torch.nn.init.xavier_uniform_(conv_layer.weight) 
            
        case 'xavier_normal':
            torch.nn.init.xavier_normal_(conv_layer.weight)  
            
        case 'uniform':
            torch.nn.init.uniform_(conv_layer.weight)     
            
        case 'normal':
            torch.nn.init.normal_(conv_layer.weight)     
            
        case _:
            raise ValueError(f"Unsupported initialization type: {initialization}.")      
        

        
        
import numpy as np
import torch

import torch
import numpy as np

def change_weights(module, SVD=False):
    
    if SVD:  
        device = torch.device('cuda')  # Using GPU

        n_channels = module.weight.shape[0]  # =n_filters (out)
        n_elements = module.weight.shape[1] * module.weight.shape[2] * module.weight.shape[3]  # =in_channels*kernel_height*kernel_width
        n_components = min(n_channels, n_elements)
        power_law_exponent = -1  # Exponent of power law decay

        # Create decomposed matrices
        U, _ = torch.linalg.qr(torch.randn(n_channels, n_components, device=device))
        V, _ = torch.linalg.qr(torch.randn(n_elements, n_components, device=device))
        V_T = V.T

        eigenvalues = torch.pow(torch.arange(1, n_components + 1, dtype=torch.float32, device=device), power_law_exponent)
        S = torch.zeros((n_components, n_components), device=device)
        
        # Correct way to fill the diagonal
        torch.diagonal(S).copy_(torch.sqrt(eigenvalues * (n_channels - 1)))

        # Calculate data with specified eigenvalues
        X = U @ S @ V_T

        # Reshape and set weights
        weights = X.reshape(n_channels, module.weight.shape[1], module.weight.shape[2], module.weight.shape[3])
        module.weight = torch.nn.Parameter(data=weights)
        return module
        
        
    else:
        
        in_size = module.weight.shape[1]
        k_size = module.weight.shape[2]
        n_channels = module.weight.shape[0]

        n_elements = in_size * k_size * k_size
        power_law_exponent = -1

        # Efficient computation of eigenvalues
        eigenvalues = np.arange(1, n_elements + 1, dtype=np.float32) ** power_law_exponent

        # Direct generation of an orthonormal matrix
        eigenvectors = torch.linalg.qr(torch.randn(n_elements, n_elements))[0]

        # Scale eigenvectors' variances
        eigenvectors = eigenvectors * torch.sqrt(torch.from_numpy(eigenvalues))[None, :]

        # Generate random data and compute weights
        X = torch.randn(n_channels, n_elements) @ eigenvectors
        weights = X.reshape(n_channels, in_size, k_size, k_size)

        # Update module weights
        module.weight = torch.nn.Parameter(weights)
        return module

        
