import os
import functools
import pickle
import logging

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

from config import setup_logging

load_dotenv()
setup_logging()
torch.manual_seed(42)

CACHE = os.getenv("CACHE")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INDICES = torch.randperm(224**2)

def cache(file_name_func):

    def decorator(func):
        
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):

            file_name = file_name_func(*args, **kwargs) 
            cache_path = os.path.join(CACHE, file_name)
            
            if os.path.exists(cache_path):
                logging.info('Loading processed images')
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            
            result = func(self, *args, **kwargs)
            with open(cache_path,'wb') as f:
                pickle.dump(result, f)
            return result

        return wrapper
    return decorator


import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import logging

class ImageProcessor:
    """
    A utility class to preprocess and transform images. It includes caching functionalities to avoid
    repetitive computations.

    Attributes:
        device (torch.device): The device to which tensors should be sent.
        batch_size (int, optional): Number of samples per batch of computation. Defaults to 100.
    """
    
    def __init__(self, device, batch_size=100):
        self.device = device
        self.batch_size = batch_size

        if not os.path.exists(os.path.join(CACHE, 'preprocessed_images')):
            os.mkdir(os.path.join(CACHE, 'preprocessed_images'))

    @staticmethod
    def cache_file(image_paths, dataset, image_size=224):
        if 'naturalscenes' in dataset:
            num_images = 73000
        else:
            num_images = len(image_paths)
        name = f'{dataset}_size={image_size}_num_images={num_images}'
        return os.path.join('preprocessed_images', name)

    @cache(cache_file)
    def process(self, image_paths, dataset, image_size=224):
        """
        Process and transform a list of images.

        Args:
            image_paths (list): List of image file paths.
            dataset (str): Dataset name.

        Returns:
            torch.Tensor: Tensor containing the processed images.
        """
        logging.info('Processing images...')

        if 'shuffled' in dataset:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                ShufflePixels(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])

        dataset = TransformDataset(image_paths, transform=transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)

        processed_images = []

        for batch in tqdm(dataloader):
            batch = batch.to("cpu")
            processed_images.append(batch)

        return torch.cat(processed_images, dim=0)



class TransformDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')  # Convert image to RGB
    
                          
        if self.transform:
            img = self.transform(img)
        
        return img


class ShufflePixels(nn.Module):
    def __init__(self):
        super(ShufflePixels, self).__init__()

    def forward(self, img):
        """
        Shuffle the pixels of an image.
        Args:
        - img (torch.Tensor): An image tensor of shape (channels, height, width).
        Returns:
        - torch.Tensor: An image with shuffled pixels.
        """
        C, H, W = img.shape
        shuffled_img = torch.empty_like(img)

        for c in range(C):
            pixels = img[c].view(-1)  # Flatten
            shuffled_pixels = pixels[INDICES]  # Shuffle
            shuffled_img[c] = shuffled_pixels.view(H, W)  # Reshape

        return shuffled_img
        
