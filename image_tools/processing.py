import torch
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
import sys
import h5py
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
from config import CACHE


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
    
   
def load_images(dataset, batch_size):

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    
    all_batches = []
    for batch in tqdm(dataloader, desc="Loading images"):
        all_batches.append(batch)
        
    return torch.cat(all_batches, dim=0)



def save_images(file_path, dataset, batch_size):
    
    dataloader = DataLoader(dataset, batch_size= batch_size, shuffle=False, num_workers=1)

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
         
        
        if os.path.exists(file_path):
            dataset = H5Dataset(file_path = file_path, device=self.device)
            return load_images(dataset=dataset, batch_size=self.batch_size)
    
        else:            
            print('processing images...')
            
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

            dataset = TransformDataset(self.image_paths, transform=transform)
            save_images(file_path=file_path, dataset=dataset, batch_size=self.batch_size)
         
        dataset = H5Dataset(file_path = file_path, device=self.device)
        return load_images(dataset=dataset, batch_size=self.batch_size)



class TransformDataset(Dataset):
    
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



class H5Dataset(Dataset):
    def __init__(self,
                file_path,
                device):  
        
        self.file_path = file_path
        self.hf_file = h5py.File(self.file_path, 'r')
        self.device=device
        self.keys = sorted(self.hf_file.keys(), key=lambda x: int(x.split('_')[-1]))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        image = torch.tensor(self.hf_file[key][:])
        return image
    
    def num_images(self):
        return sum(self.hf_file[key].shape[0] for key in self.hf_file.keys())
        

        
