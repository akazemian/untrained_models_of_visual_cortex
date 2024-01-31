import sys
import torchvision
import torch
from torch import nn
import pickle
import os
from model_features.layer_operations.convolution import Convolution, initialize_conv_layer
import timm
from timm.models.vision_transformer import VisionTransformer
from model_features.layer_operations.output import Output
torch.manual_seed(0)
torch.cuda.manual_seed(0)


# load untrained mdoel
untrained_model = timm.create_model('vit_base_patch16_224', pretrained=False).cuda()
EMBED_DIM = untrained_model.blocks[0].attn.qkv.in_features  
OUT_FEATURES = untrained_model.blocks[0].mlp.fc1.out_features
MLP_RATIO = int(OUT_FEATURES/EMBED_DIM)  
NUM_LAYERS = len(untrained_model.blocks) 
NUM_HEADS = untrained_model.blocks[0].attn.num_heads 


class TimmViT(nn.Module):
    def __init__(self, use_wavelets:bool, out_features:int):
        super(TimmViT, self).__init__()

        self.out_features = out_features
        self.use_wavelets = use_wavelets
        
        in_channels = 108 if self.use_wavelets else 3
        
        self.model =  VisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=in_channels,
            embed_dim=OUT_FEATURES,
            depth=len(untrained_model.blocks),  # more layers
            num_heads=NUM_HEADS,  # more attention heads
            mlp_ratio=int(OUT_FEATURES/EMBED_DIM)  ,  # higher mlp ratio
            num_classes=1000  
        )
        

    def forward(self, x):
        return self.model(x)    
    
    
    
class BaseModel(nn.Module):
    def __init__(self, out_features:int, block:int, use_wavelets:bool, 
                 conv: nn.Module, last:nn.Module, 
                 device:str='cuda'):
        super(BaseModel, self).__init__()

        self.block = block
        self.out_features = out_features
        self.use_wavelets = use_wavelets
        
        self.conv = conv
        self.last = last
        
        self.model = TimmViT(out_features = self.out_features)
        self.device = device
        
        self.activations = {}
        self.register_hooks()

    
    def _hook_fn(self, idx, module, input, output):
        """ Hook function to capture activations without the class token. """
        # Remove the class token (first token) from the output
        activations_without_class_token = output[:, 1:, :]  # Exclude the first token
        self.activations[f'block_{idx}'] = activations_without_class_token

    def register_hooks(self):
        # Register hooks on each transformer block with the correct index
        for i, blk in enumerate(self.model.model.blocks):
            blk.register_forward_hook(lambda m, inp, out, idx=i: self._hook_fn(idx, m, inp, out))

    def forward(self, x):
        x = x.to(self.device)
        self.model.to(self.device)
        
        if self.use_wavelets:
            x = self.conv(x)
        
        _ = self.model(x)  # Perform the forward pass to populate activations

        # Use activations from a specific layer
        activations = self.activations.get(f'block_{self.block}', None)
        if activations is not None:
            x = self.last(activations)
        else:
            raise ValueError(f"No activations found for block {self.block}")

        return x
    
    
class ViTBase:
    
    def __init__(self, out_features:int, use_wavelets:bool=False, block=11, device='cuda'):
    
        self.block = block
        self.out_features = out_features
        self.use_wavelets = use_wavelets
        self.device = device
        
        filter_params = {'type':'curvature','n_ories':12,'n_curves':3,'gau_sizes':(5,),'spatial_fre':[1.2]},
        filters_1 = filter_params['n_ories']*filter_params['n_curves']*len(filter_params['gau_sizes']*len(filter_params['spatial_fre']))*3
        
    def Build(self):
    
        conv = Convolution(filter_size=15, filter_params=self.filter_params)     
        last = Output()
        
        return BaseModel(out_features = self.out_features,
                         block=self.block,
                         use_wavelets = self.use_wavelets,
                         conv = conv,
                         last = last)
    
    
    
    
    
    
    
