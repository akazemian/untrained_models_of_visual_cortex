import torch
from PIL import Image
from torchvision import transforms
import torch
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
    

def preprocess(images, size=224):
                
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        transform = transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

        dataset = CustomImageDataset(images, transform=transform)
        dataloader = DataLoader(dataset, shuffle=False, num_workers=1)

        processed_images_list = []

        for batch in dataloader:
            batch = batch.to(device) 
            processed_images_list.append(batch)
     
        return torch.cat(processed_images_list, 0)
    


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






        
