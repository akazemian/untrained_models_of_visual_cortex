import math
from typing import Optional, Tuple
from torch import nn
import torch
from code_.model_activations.models.layer_operations.convolution import WaveletConvolution, initialize_conv_layer
from code_.model_activations.models.layer_operations.output import Output
from code_.model_activations.models.layer_operations.nonlinearity import NonLinearity

torch.manual_seed(42)
torch.cuda.manual_seed(42)

class Model(nn.Module):
    """Expansion model architecture consisting of 5 convolutional and pooling layers."""
    def __init__(self, conv1: nn.Module, pool1: nn.Module, conv2: nn.Module, pool2: nn.Module, 
                 conv3: nn.Module, pool3: nn.Module, conv4: nn.Module, pool4: nn.Module, 
                 conv5: nn.Module, pool5: nn.Module, nl: nn.Module, last: nn.Module, device: str) -> None:
        super(Model, self).__init__()
        self.conv1 = conv1
        self.pool1 = pool1
        self.conv2 = conv2
        self.pool2 = pool2
        self.conv3 = conv3
        self.pool3 = pool3
        self.conv4 = conv4
        self.pool4 = pool4
        self.conv5 = conv5
        self.pool5 = pool5
        self.nl = nl
        self.last = last
        self.device = device
        
    def forward(self, x)-> torch.Tensor:
        """Forward pass through the network, applying each layer sequentially."""
        x = x.to(self.device)
        # Processing layers 1 to 4
        for i in range(1, 5):
            conv = getattr(self, f'conv{i}')
            pool = getattr(self, f'pool{i}')
            x = conv(x)
            x = self.nl(x)
            x = pool(x)
        
        #layer 5
        i = 5
        conv = getattr(self, f'conv{i}')
        pool = getattr(self, f'pool{i}')
        
        _, _, h, w = x.shape
        
        # Shuffle along the 3rd dimension (height)
        shuffle_h = torch.randperm(h)
        x_shuffled_h = x[:, :, shuffle_h, :]
        
        # Shuffle along the 4th dimension (width)
        shuffle_w = torch.randperm(w)
        x_shuffled = x_shuffled_h[:, :, :, shuffle_w]
        
        x = conv(x_shuffled)
        x = self.nl(x)
        x = pool(x)
        
        # Output layer
        x = self.last(x)
        return x

class ExpansionNoSP:
    """Constructing the 5-layer expansion model with customizable filter sizes and types."""
    def __init__(self, filters_1: Optional[int] = None, filters_2: int = 1000, filters_3: int = 3000, 
                 filters_4: int = 5000, filters_5: int = 30000, init_type: str = 'kaiming_uniform', 
                 non_linearity: str = 'relu', device: str = 'cuda') -> None:
        self.filters_1 = filters_1
        self.filters_2 = filters_2
        self.filters_3 = filters_3
        self.filters_4 = filters_4
        self.filters_5 = filters_5
        self.init_type = init_type
        self.non_linearity = non_linearity
        self.device = device

    def create_layer(self, in_filters: int, out_filters: int, kernel_size: Tuple[int, int], 
                     stride: int = 1, pool_kernel: int = 2, pool_stride: Optional[int] = None, 
                     padding: int = 0) -> Tuple[nn.Module, nn.Module]:
        """Creates a convolutional layer and a pooling layer with either fixed or random conv filters"""
        conv = nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size,
                         stride=stride, padding=padding, bias=False).to(self.device)
        initialize_conv_layer(conv, self.init_type)
        pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride)
        return conv, pool

    def build(self):
        """Builds the complete model using specified configurations."""
        # Pre-fixed or random filters for layer 1
        if self.filters_1 is None:
            conv1 = WaveletConvolution(filter_size=15, filter_type='curvature', device=self.device)
            pool1 = nn.AvgPool2d(kernel_size=2)
            self.filters_1 = conv1.layer_size
        else:
            padding = math.floor(15 / 2)
            conv1, pool1 = self.create_layer(3, self.filters_1, (15, 15), padding=padding)

        # Setup layers 2 to 5
        conv2, pool2 = self.create_layer(self.filters_1, self.filters_2, (7, 7), 1, 2)
        conv3, pool3 = self.create_layer(self.filters_2, self.filters_3, (5, 5), 1, 2)
        conv4, pool4 = self.create_layer(self.filters_3, self.filters_4, (3, 3), 1, 2)
        conv5, pool5 = self.create_layer(self.filters_4, self.filters_5, (3, 3), 1, 4, 1)

        nl = NonLinearity(self.non_linearity)
        last = Output()

        return Model(
            conv1, pool1, conv2, pool2, conv3, pool3, conv4, pool4, conv5, pool5, nl, last, self.device
        )
