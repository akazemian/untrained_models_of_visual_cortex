import torch
from PIL import Image
from torchvision import transforms
import torch
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import functools
import os
from joblib import dump, load
from tools import cache
import sys
from joblib import Memory
import h5py
import numpy as np

sys.path.append(os.getenv('BONNER_ROOT_PATH'))
from config import CACHE
memory = Memory(CACHE, verbose=0, mmap_mode='r')

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
    



import functools
import os
from joblib import dump, load
from config import CACHE


    
class H5Dataset():
    def __init__(self,
                file_path,
                device):
        
        self.file_path = file_path
        self.hf_file = h5py.File(self.file_path, 'r')
        self.device=device
        
        
    def __len__(self):
        return sum(self.hf_file[key].shape[0] for key in self.hf_file.keys())

    
    def load_images(self):
        print('loading images...')
        batches = [self.hf_file[key][:] for key in tqdm(sorted(self.hf_file.keys(), key=lambda x: int(x.split('_')[-1])))]
        return np.concatenate(batches, axis=0)        

    

def save_images(file_path, dataloader):
    mode = 'a' if os.path.exists(file_path) else 'w'
    with h5py.File(file_path, mode) as hf:
            existing_batches = len(hf.keys())
            for batch_idx, batch in enumerate(tqdm(dataloader), start=existing_batches):
                hf.create_dataset(f'batch_{batch_idx}', data=batch.cpu().numpy())
    return



class ImageProcessor:
    def __init__(self, image_paths, dataset, image_size, batch_size, device):
        
        self.batch_size = batch_size
        self.image_paths = image_paths
        self.dataset = dataset
        self.image_size = image_size
        self.device = device
        
        if not os.path.exists(os.path.join(CACHE,'preprocessed_images')):
            os.mkdir(os.path.join(CACHE,'preprocessed_images'))

    def preprocess(self):

        name = f'dataset={self.dataset}_size={self.image_size}_number_of_images={len(self.image_paths)}'
        file_path = os.path.join(CACHE,'preprocessed_images',name)
        h5data = H5Dataset(file_path = file_path, device = self.device) 
        
        if os.path.exists(file_path) and h5data.__len__()==len(self.image_paths):
            return h5data.load_images()
    
        else:            
            print('processing images...')
            
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

            dataset_instance = CustomImageDataset(self.image_paths, transform=transform)
            dataloader = DataLoader(dataset_instance, batch_size= self.batch_size, shuffle=False, num_workers=1)
            save_images(file_path, dataloader)
         
        return load_all_images(file_path)

    


class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return img






        
