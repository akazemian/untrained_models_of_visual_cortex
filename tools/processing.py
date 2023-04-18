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
    
ROOT = os.getenv('MB_DATA_PATH')
IMAGES_PATH = os.path.join(ROOT,'processed_images')
print()

def get_path(images, dataset, file_ext):
    
    
    if dataset == 'naturalscenes_zscored_processed':
        # shared images case
        if len(images) == 872:
            p = f"{IMAGES_PATH}/{file_ext}_{dataset}_shared.npy"
    
        #unshared images case
        elif len(images) == (73000-872):
            p = f"{IMAGES_PATH}/{file_ext}_{dataset}_unshared.npy"
    
        # sub sample of unshared images case
        elif len(images) == 10000:
            p = f"{IMAGES_PATH}/{file_ext}_{dataset}_subset.npy"
        
        # all images case
        elif len(images) == 73000:
            p = f"{IMAGES_PATH}/{file_ext}_{dataset}.npy" 
        
        else:
            raise ValueError('The number of images does not match any of the cases')  
    
    else:
        p = None
    
    return p
    

    

def PreprocessGSSmall(images, dataset, size = 96):
    
    file_ext = 'preprocessgs'
    processed_img_path = get_path(images, dataset, file_ext)
    
    if os.path.exists(processed_img_path):
        print('loading processed images...')
        return np.load(processed_img_path,mmap_mode='r+') 
    
    print('processing images...')
    transform = transforms.Compose([
         transforms.Resize((size,size)),
         transforms.Grayscale(),
         transforms.ToTensor(),
         transforms.Normalize(mean=0.5, std=0.5)])
    
    try:
        processed_images = np.stack([transform(Image.open(i).convert('RGB')) for i in images])
    except: 
        processed_images = torch.stack([transform(Image.open(i).convert('RGB')) for i in images])
    
    np.save(processed_img_path,processed_images)
    return processed_images
  


    

def PreprocessGSLarge(images, dataset, size = 224):
    
    file_ext = 'preprocessgs'
    processed_img_path = get_path(images, dataset, file_ext)
    
    if os.path.exists(processed_img_path):
        print('loading processed images...')
        return np.load(processed_img_path,mmap_mode='r+') 
    
    print('processing images...')
    transform = transforms.Compose([
         transforms.Resize((size,size)),
         transforms.Grayscale(),
         transforms.ToTensor(),
         transforms.Normalize(mean=0.5, std=0.5)])
    
    try:
        processed_images = np.stack([transform(Image.open(i).convert('RGB')) for i in images])
    except: 
        processed_images = torch.stack([transform(Image.open(i).convert('RGB')) for i in images])
    
    np.save(processed_img_path,processed_images)
    return processed_images


    
    

def PreprocessRGBSmall(images, dataset, size = 96):
    
    file_ext = 'preprocesmodelrgb'
    processed_img_path = get_path(images, dataset, file_ext)
    
    if os.path.exists(processed_img_path):
        print('loading processed images...')
        return np.load(processed_img_path,mmap_mode='r+') 
    
    print('processing images...')
    
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
    
    np.save(processed_img_path,processed_images)
    return processed_images

    
    
    
    
def PreprocessRGBLarge(images, dataset, size = 224):
    
    file_ext = 'preprocesrgb'
    processed_img_path = get_path(images, dataset, file_ext)

    if os.path.exists(processed_img_path):
        print('loading processed images...')
        return np.load(processed_img_path,mmap_mode='r+') 
    
    print('processing images...')
    
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

    np.save(processed_img_path,processed_images)
    return processed_images


    


        
