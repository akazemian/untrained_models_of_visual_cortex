import sys
import torchvision
import torch
from torch import nn
import pickle
import os
import timm
from timm.models.vision_transformer import VisionTransformer
from model_features.layer_operations.output import Output
torch.manual_seed(0)
torch.cuda.manual_seed(0)



# base model
# model = timm.create_model('vit_base_patch16_224', pretrained=False).cuda()
# model.head = torch.nn.Identity()


# base model with larger embedding dim
# reference_model = timm.create_model('vit_base_patch16_224', pretrained=False)
# embed_dim = reference_model.blocks[0].attn.qkv.in_features  # Double the typical base model's embed_dim
# num_heads = reference_model.blocks[0].attn.num_heads #Double the typical base model's num_heads
# out_features = reference_model.blocks[0].mlp.fc1.out_features
# mlp_ratio = int(out_features/embed_dim)   # Double the typical base model's mlp_ratio
# num_layers = len(reference_model.blocks) #     # Typical number of layers for a base model

# model = VisionTransformer(
#     img_size=224,
#     patch_size=16,
#     embed_dim=embed_dim*4,
#     depth=num_layers,
#     num_heads=num_heads,
#     mlp_ratio=mlp_ratio,
#     num_classes=1000  
# )
# model.head = torch.nn.Identity()




# model with larger hidden dims, learned positional encodings and large linear head
reference_model = timm.create_model('vit_base_patch16_224', pretrained=False)
embed_dim = reference_model.blocks[0].attn.qkv.in_features  # Double the typical base model's embed_dim
num_heads = reference_model.blocks[0].attn.num_heads #Double the typical base model's num_heads
out_features = reference_model.blocks[0].mlp.fc1.out_features
mlp_ratio = int(out_features/embed_dim)   # Double the typical base model's mlp_ratio
num_layers = len(reference_model.blocks) #     # Typical number of layers for a base model


model = VisionTransformer(
    img_size=224,
    patch_size=16,
    embed_dim=embed_dim,
    depth=num_layers,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio*4,
    num_classes=1000  
)
pretrained_model = timm.create_model('vit_base_patch16_224', pretrained=True)

model.pos_embed = pretrained_model.pos_embed
model.head = torch.nn.Linear(embed_dim, 108000)





class Model(nn.Module):
    
    
    def __init__(self,
                last:nn.Module,
                ):
        
        super(Model, self).__init__()
        

        self.last = last
        
        
    def forward(self, x):
                
        x = x.cuda()
        model.cuda()
        x = model(x)
        x = self.last(x)
        print(x.shape)
        
        return x    


    
    
    
class ViT:

    
    def __init__(self,device='cuda'):
    
        self.device = device
        
    def Build(self):
    
        last = Output()
        
        return Model(
                last = last)