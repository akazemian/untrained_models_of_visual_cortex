import sys
import torchvision
import torch
from torch import nn
import pickle
import os
import timm
#from timm.models.vision_transformer import VisionTransformer
from layer_operations.output import Output
torch.manual_seed(0)
torch.cuda.manual_seed(0)


# load untrained mdoel
untrained_model = timm.create_model('vit_base_patch16_224', pretrained=False).cuda()
EMBED_DIM = untrained_model.blocks[0].attn.qkv.in_features  
OUT_FEATURES = untrained_model.blocks[0].mlp.fc1.out_features
MLP_RATIO = int(OUT_FEATURES/EMBED_DIM)  
NUM_LAYERS = len(untrained_model.blocks) 
NUM_HEADS = untrained_model.blocks[0].attn.num_heads 


class CustomAttention(nn.Module):
    def __init__(self, in_chans, embed_dim, num_heads):
        super(CustomAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5

        self.in_chans = in_chans
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout(p=0.0)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj_drop = nn.Dropout(p=0.0)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) 
        q, k = self.q_norm(q), self.k_norm(k)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CustomMLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio), bias=True)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(p=0.0)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim, bias=True)
        self.drop2 = nn.Dropout(p=0.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class CustomViTBlock(nn.Module):
    def __init__(self, in_chans, new_embed_dim, num_heads, mlp_ratio):
        super(CustomViTBlock, self).__init__()

        self.norm1 = nn.LayerNorm(new_embed_dim, eps=1e-06)
        self.attn = CustomAttention(in_chans, new_embed_dim, num_heads)
        self.ls1 = nn.Identity()
        self.drop_path1 = nn.Identity()
        self.norm2 = nn.LayerNorm(new_embed_dim, eps=1e-06)
        self.mlp = CustomMLP(new_embed_dim, mlp_ratio)
        self.ls2 = nn.Identity()
        self.drop_path2 = nn.Identity()

    def forward(self, x):
        y = self.norm1(x)
        y = self.attn(y)
        x = x + self.drop_path1(self.ls1(y))
        y = self.norm2(x)
        y = self.mlp(y)
        x = x + self.drop_path2(self.ls2(y))
        return x
        


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.Identity()

    def forward(self, x):
        x = self.proj(x)  # [B, E, H, W]
        x = x.flatten(2)  # [B, E, N]
        x = x.transpose(1, 2)  # [B, N, E]
        print('patch embed shape',x.shape)
        return x
        
        
class VisTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, new_embed_dim=1024):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([])
        for i in range(depth):
            if i < 10:
                self.blocks.append(CustomViTBlock(in_chans, embed_dim, num_heads, mlp_ratio))
            
            elif i == 10:
                self.blocks.append(CustomViTBlock(in_chans, embed_dim, num_heads, mlp_ratio))
                self.blocks.append(nn.Sequential(
                    nn.Linear(embed_dim, new_embed_dim),
                    nn.ReLU()))
            else:
                self.blocks.append(CustomViTBlock(in_chans, new_embed_dim, num_heads, mlp_ratio))
            
    

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)
            print('block output shape',x.shape)

        return x


    
class CustomModel(nn.Module):
    def __init__(self, block:int, 
                 vit:nn.Module, 
                 last:nn.Module, 
                 device:str='cuda'):
        super(CustomModel, self).__init__()

        self.block = block
        self.vit = vit
        self.last = last

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
        for i, blk in enumerate(self.vit.blocks):
            blk.register_forward_hook(lambda m, inp, out, idx=i: self._hook_fn(idx, m, inp, out))

    def forward(self, x):
        x = x.to(self.device)
        self.vit.to(self.device)
    
        _ = self.vit(x)  # Perform the forward pass to populate activations

        # Use activations from a specific layer
        activations = self.activations.get(f'block_{self.block}', None)
        if activations is not None:
            x = self.last(activations)
        else:
            raise ValueError(f"No activations found for block {self.block}")

        print('final output shape',x.shape)
        return x
    



class CustomViT:
    
    def __init__(self, out_features:int, block=11, device='cuda'):
    
        self.block = block
        self.out_features = out_features
        self.device = device
                
    def Build(self):
            
        vit = VisTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=EMBED_DIM,
            depth=len(untrained_model.blocks),  # more layers
            num_heads=NUM_HEADS,  # more attention heads
            mlp_ratio=int(OUT_FEATURES/EMBED_DIM),
            new_embed_dim= self.out_features
        )
        
        last = Output()
        
        return CustomModel(block=self.block,
                         vit = vit,
                         last = last)










    
    
    
