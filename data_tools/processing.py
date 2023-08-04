import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from random import sample,seed
from .config import PROCESSED_IMAGES_PATH

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
    

def preprocess(images, dataset, size=224):
                
        if not os.path.exists(PROCESSED_IMAGES_PATH):
            os.mkdir(PROCESSED_IMAGES_PATH)
            
        file_ext = f'preproces_rgb_{size}'
        processed_img_path = f"{PROCESSED_IMAGES_PATH}/{file_ext}_{dataset}_{len(images)}.npy"

        if os.path.exists(processed_img_path) and processed_img_path is not None:
            print('loading processed images...')
            return torch.Tensor(np.load(processed_img_path,mmap_mode='r+')) 

        print('processing images...')

        transform = transforms.Compose([
            transforms.Resize((self.im_size,self.im_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

        try:
            processed_images = np.stack([transform(Image.open(i).convert('RGB')) for i in images])
        except: 
            processed_images = torch.stack([transform(Image.open(i).convert('RGB')) for i in images])


        np.save(processed_img_path,processed_images)
        return torch.Tensor(processed_images)
    


    


        
