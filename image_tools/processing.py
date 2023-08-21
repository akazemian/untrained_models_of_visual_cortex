from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import os
import sys
import os 
import sys
import functools
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pickle

sys.path.append(os.getenv('BONNER_ROOT_PATH'))
from config import CACHE


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)



def cache(file_name_func):

    def decorator(func):
        
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):

            file_name = file_name_func(*args, **kwargs) 
            cache_path = os.path.join(CACHE, file_name)
            
            if os.path.exists(cache_path):
                print('loading processed images...')
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            
            result = func(self, *args, **kwargs)
            with open(cache_path,'wb') as f:
                pickle.dump(result, f)
            return result

        return wrapper
    return decorator




class ImageProcessor:
    """
    A utility class to preprocess and transform images. It includes caching functionalities to avoid
    repetitive computations.

    Attributes:
        device (torch.device): The device to which tensors should be sent.
        batch_size (int, optional): Number of samples per batch of computation. Defaults to 100.
    """
    
    def __init__(self, device, batch_size = 100):
                
        self.device = device
        self.batch_size = batch_size
        self.im_size = 224
        self.transform = transforms.Compose([
            transforms.Resize((self.im_size, self.im_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
  

        
        if not os.path.exists(os.path.join(CACHE,'preprocessed_images')):
            os.mkdir(os.path.join(CACHE,'preprocessed_images'))
        
        
    @staticmethod
    def cache_file(image_paths, dataset):
        name = f'{dataset}_num_images={len(image_paths)}'
        return os.path.join('preprocessed_images',name)

    
    @cache(cache_file)
    def process(self, image_paths, dataset):        
        """
        Process and transform a list of images.

        Args:
            image_paths (list): List of image file paths.
            dataset (str): Dataset name.

        Returns:
            torch.Tensor: Tensor containing the processed images.
        """
        print('processing images...')
        dataset = TransformDataset(image_paths, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        return torch.cat([batch for batch in tqdm(dataloader)],dim=0)
    

    def process_batch(self, image_paths, dataset):
        """
        Process a batch of images without using cache.

        Args:
            image_paths (list): List of image file paths.
            dataset (str): Dataset name.

        Returns:
            torch.Tensor: Tensor containing the processed images.
        """
        dataset = TransformDataset(image_paths, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        return torch.cat([batch for batch in dataloader],dim=0)




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
        
