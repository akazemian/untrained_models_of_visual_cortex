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
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
    
    
    

class Preprocess:
    
    def __init__(self,im_size):
        self.im_size = im_size
        
    
    def PreprocessGS(self, images, dataset):
    
        file_ext = f'preprocess_gs_{self.im_size}'
        processed_img_path = f"{IMAGES_PATH}/{file_ext}_{dataset}_{len(images)}.npy"

        if os.path.exists(processed_img_path) and processed_img_path is not None:
            print('loading processed images...')
            return torch.Tensor(np.load(processed_img_path,mmap_mode='r+'))

        print('processing images...')
        transform = transforms.Compose([
             transforms.Resize((self.im_size,self.im_size)),
             transforms.Grayscale(),
             transforms.ToTensor(),
             transforms.Normalize(mean=0.5, std=0.5)])

        try:
            processed_images = np.stack([transform(Image.open(i).convert('RGB')) for i in images])
        except: 
            processed_images = torch.stack([transform(Image.open(i).convert('RGB')) for i in images])

        np.save(processed_img_path,processed_images)
        return torch.Tensor(processed_images)
  

    

    def PreprocessRGB(self, images, dataset):

        file_ext = f'preproces_rgb_{self.im_size}'
        processed_img_path = f"{IMAGES_PATH}/{file_ext}_{dataset}_{len(images)}.npy"

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
    


    


        
