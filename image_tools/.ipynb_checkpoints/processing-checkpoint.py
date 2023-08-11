from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import os
import sys
device = 'cuda'

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)



def processor(image_paths, image_size):


        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

                                
        return torch.stack([transform(Image.open(i).convert('RGB')).to(device) for i in image_paths])


