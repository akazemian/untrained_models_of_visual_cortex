from torchvision import transforms
from PIL import Image
import numpy as np
import torch
from random import sample,seed
import os
#import logging
import numpy as np
import h5py
from PIL import Image
import xarray as xr
import pandas as pd



def LoadObject2VecImages():
    data_path = '/data/shared/datasets/object2vec/stimuli'
    all_images = []
    for folder in os.listdir(data_path):
        for image in os.listdir(f'{data_path}/{folder}'):
            all_images.append(f'{data_path}/{folder}/{image}')
    return all_images    



def LoadImagenet21kImages(num_classes=1000,num_per_class=50):
    seed(0)
    all_images = []
    #path = '/data/shared/datasets/imagenet21k_sorscher2021'
    path = '/data/shared/datasets/ilsvrc2012/train'
    folders = os.listdir(path)
    cats = sample(folders,num_classes)
    for cat in cats:
        images = os.listdir(os.path.join(path,cat))
        examples = sample(images,num_per_class)
        for example in examples:
            example_path = os.path.join(os.path.join(path,cat,example))
            all_images.append(example_path)
    return all_images
    
    
    
    
def LoadNSDImages(shared_images=False):
    
    path = '/data/shared/datasets/allen2021.natural_scenes/images'
    all_images = []
    

    print('shared images:',shared_images)
    for image in sorted(os.listdir(path)):
        if shared_images:
            shared_ids = list(xr.open_dataset('/data/atlas/activations/model_final_100_naturalscenes').stimulus_id.values)
            if image.strip('.png') in shared_ids:
                all_images.append(f'{path}/{image}')
            
        else:
            all_images.append(f'{path}/{image}')

    return all_images 
    

        
def LoadMajajHongImages():
    
    all_images = []
    path = '/data/shared/brainio/brain-score/dicarlo.hvm-public'
    for image in sorted(os.listdir(path)):
        all_images.append(f'{path}/{image}')
    return all_images

    
    
def LoadImagenet21kVal(num_classes=1500, num_per_class=20, separate_classes=False):
    #_logger = logging.getLogger(fullname(LoadImagenet21kVal))
    base_indices = np.arange(num_per_class).astype(int)
    indices = []
    for i in range(num_classes):
        indices.extend(50 * i + base_indices)

    framework_home = os.path.expanduser(os.getenv('MT_HOME', '~/.model-tools'))
    imagenet_filepath = os.getenv('MT_IMAGENET_PATH', os.path.join(framework_home, 'imagenet2012.hdf5'))
    imagenet_dir = f"{imagenet_filepath}-files"
    os.makedirs(imagenet_dir, exist_ok=True)

    if not os.path.isfile(imagenet_filepath):
        os.makedirs(os.path.dirname(imagenet_filepath), exist_ok=True)
        #_logger.debug(f"Downloading ImageNet validation to {imagenet_filepath}")
        s3.download_file("imagenet2012-val.hdf5", imagenet_filepath)

    filepaths = []
    with h5py.File(imagenet_filepath, 'r') as f:
        for index in indices:
            imagepath = os.path.join(imagenet_dir, f"{index}.png")
            if not os.path.isfile(imagepath):
                image = np.array(f['val/images'][index])
                Image.fromarray(image).save(imagepath)
            filepaths.append(imagepath)

    if separate_classes:
        filepaths = [filepaths[i * num_per_class:(i + 1) * num_per_class]
                    for i in range(num_classes)]
    return filepaths


def LoadLocalizerImages():
    path = '/home/akazemi3/Desktop/localizer_stimuli'
    all_images = []
    
    for image in sorted(os.listdir(path)):
        all_images.append(f'{path}/{image}')
    return all_images    
    
def LoadImagePaths(name,**kwargs): #num_classes=None,num_per_class=None,shared_images=True):
    
    if name == 'object2vec':
        return LoadObject2VecImages()
        
    elif name == 'imagenet21k':
        return LoadImagenet21kImages(**kwargs)

    elif name == 'imagenet21k_val':
        #num_classes=1000
        #num_per_class=10
        return LoadImagenet21kVal(kwargs)
    
    elif 'naturalscenes' in name:
        return LoadNSDImages(**kwargs)
    
    elif name == 'majajhong':
        return LoadMajajHongImages()
    
    elif name == 'localizers':
        return LoadLocalizerImages()
    
    
    
def get_image_labels(dataset,images):
    if 'majajhong' in dataset:
        name_dict = pd.read_csv('/data/shared/brainio/brain-score/image_dicarlo_hvm-public.csv').set_index('image_file_name')['image_id'].to_dict()
        return [name_dict[os.path.basename(i)] for i in images]
    

    if 'naturalscenes' in dataset:
        return [os.path.basename(i).strip('.png') for i in images]