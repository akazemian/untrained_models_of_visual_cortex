from torchvision import transforms
from PIL import Image
import numpy as np
import torch
from random import sample,seed
import os
import logging
import numpy as np
import h5py
from PIL import Image
    



def PreprocessGS(images,dataset):
    
    if dataset == 'naturalscenes_zscored_processed' and len(images) == 872:
        PROCESSED_IMG_PATH = f"/data/atlas/processed_images/preprocessgs_{dataset}_shared.npy"
    
    elif dataset == 'naturalscenes_zscored_processed' and len(images) == (73000-872):
        PROCESSED_IMG_PATH = f"/data/atlas/processed_images/preprocessgs_{dataset}_unshared.npy"
    
    elif dataset == 'naturalscenes_zscored_processed' and len(images) == 10000:
        PROCESSED_IMG_PATH = f"/data/atlas/processed_images/preprocessgs_{dataset}_subset.npy"
        
    else:
        PROCESSED_IMG_PATH = f"/data/atlas/processed_images/preprocessgs_{dataset}.npy"
    
    if os.path.exists(PROCESSED_IMG_PATH):
        print('loading processed images...')
        return np.load(PROCESSED_IMG_PATH,mmap_mode='r+') 
    
    print('processing images...')
    size = 96
    transform = transforms.Compose([
         transforms.Resize((size,size)),
         transforms.Grayscale(),
         transforms.ToTensor(),
         transforms.Normalize(mean=0.5, std=0.5)])
    
    try:
        processed_images = np.stack([transform(Image.open(i).convert('RGB')) for i in images])
    except: 
        processed_images = torch.stack([transform(Image.open(i).convert('RGB')) for i in images])
    
    np.save(PROCESSED_IMG_PATH,processed_images)
    return processed_images
  

    

def PreprocessModelRGB(images,dataset):
    
    
    if dataset == 'naturalscenes_zscored_processed' and len(images) == 872:
        PROCESSED_IMG_PATH = f"/data/atlas/processed_images/preprocesmodelrgb_{dataset}_shared.npy"
    
    elif dataset == 'naturalscenes_zscored_processed' and len(images) == (73000-872):
        PROCESSED_IMG_PATH = f"/data/atlas/processed_images/preprocesmodelrgb_{dataset}_unshared.npy"     
    
    elif dataset == 'naturalscenes_zscored_processed' and len(images) == 10000:
        PROCESSED_IMG_PATH = f"/data/atlas/processed_images/preprocesmodelrgb_{dataset}_subset.npy"
    else:
        PROCESSED_IMG_PATH = f"/data/atlas/processed_images/preprocessmodelrgb_{dataset}.npy"
    
    if os.path.exists(PROCESSED_IMG_PATH):
        print('loading processed images...')
        return np.load(PROCESSED_IMG_PATH,mmap_mode='r+') 
    
    print('processing images...')
    size = 96
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
        
    transform = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])
    
    try:
        processed_images = np.stack([transform(Image.open(i).convert('RGB')) for i in images])
    except: 
        processed_images = torch.stack([transform(Image.open(i).convert('RGB')) for i in images])
    
    np.save(PROCESSED_IMG_PATH,processed_images)
    return processed_images

    
    
    
def PreprocessRGB(images,dataset):
    
    
    if dataset == 'naturalscenes_zscored_processed' and len(images) == 872:
        PROCESSED_IMG_PATH = f"/data/atlas/processed_images/preprocesrgb_{dataset}_shared.npy"
    
    elif dataset == 'naturalscenes_zscored_processed' and len(images) == (73000-872):
        PROCESSED_IMG_PATH = f"/data/atlas/processed_images/preprocesrgb_{dataset}_unshared.npy"  
    
    elif dataset == 'naturalscenes_zscored_processed' and len(images) == 10000:
        PROCESSED_IMG_PATH = f"/data/atlas/processed_images/preprocesrgb_{dataset}_subset.npy"
        
    else:
        PROCESSED_IMG_PATH = f"/data/atlas/processed_images/preprocessrgb_{dataset}.npy"
    
    if os.path.exists(PROCESSED_IMG_PATH):
        print('loading processed images...')
        return np.load(PROCESSED_IMG_PATH,mmap_mode='r+') 
    
    print('processing images...')
    size = 224
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])

    try:
        processed_images = np.stack([transform(Image.open(i).convert('RGB')) for i in images])
    except: 
        processed_images = torch.stack([transform(Image.open(i).convert('RGB')) for i in images])

    np.save(PROCESSED_IMG_PATH,processed_images)
    return processed_images


    


        