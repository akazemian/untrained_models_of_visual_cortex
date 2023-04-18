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
import pickle 
import random


ROOT = os.getenv('MB_DATA_PATH')

PATH_TO_NSD_SHARED_IDS = os.path.join(ROOT,'neural_data/nsd_shared_ids')
PATH_TO_NSD_SAMPLE_IDS = os.path.join(ROOT,'neural_data/nsd_sample_ids_10000')
NSD_PATH = os.path.join(ROOT,'datasets/allen2021.natural_scenes/images')
MAJAJHONG_PATH = os.path.join(ROOT,'datasets/dicarlo.hvm-public')
IMAGENET_21K_PATH = os.path.join(ROOT,'datasets/ilsvrc2012/train')





def generate_nsd_sample(num_samples=10000):
    p = LoadNSDImages(unshared_images=True)
    p_sample = random.sample(p,num_samples)
    f = open(PATH_TO_NSD_SAMPLE_IDS,'wb')
    pickle.dump(p_sample,f)
    f.close()
    return



def LoadObject2VecImages():
    
    all_images = []
    for folder in os.listdir(OBJECT_2_VEC_PATH):
        for image in os.listdir(f'{data_path}/{folder}'):
            all_images.append(f'{data_path}/{folder}/{image}')
    return all_images    




def LoadImagenet21kImages(num_classes=1000,num_per_class=5):
    
    seed(0)
    all_images = []
    folders = os.listdir(IMAGENET_21K_PATH)
    cats = sample(folders,num_classes)
    for cat in cats:
        images = os.listdir(os.path.join(IMAGENET_21K_PATH,cat))
        examples = sample(images,num_per_class)
        for example in examples:
            example_path = os.path.join(os.path.join(IMAGENET_21K_PATH,cat,example))
            all_images.append(example_path)
    return all_images
    
    
    
    
def LoadNSDImages(shared_images=False,unshared_images=False,subset=False):
     
    print('shared images:',shared_images)
    print('unshared images:',unshared_images)
    print('subset images:',subset)


    all_images = []
           
    if subset: 
        if not os.path.exists(PATH_TO_NSD_SAMPLE_IDS):
            generate_nsd_sample()
        return pickle.load(open(PATH_TO_NSD_SAMPLE_IDS,'rb'))

    else:
        
        shared_ids = pickle.load(open(PATH_TO_NSD_SHARED_IDS, 'rb'))   
        
        for image in sorted(os.listdir(NSD_PATH)):
                
            if shared_images:
                if image.strip('.png') in shared_ids:
                    all_images.append(f'{NSD_PATH}/{image}')
            
            elif unshared_images:
                if image.strip('.png') not in shared_ids:
                    all_images.append(f'{NSD_PATH}/{image}')            
            
            else:
                all_images.append(f'{NSD_PATH}/{image}')

    return all_images 
    

   

        
def LoadMajajHongImages():
    
    all_images = []
    for image in sorted(os.listdir(MAJAJHONG_PATH)):
        all_images.append(f'{MAJAJHONG_PATH}/{image}')
    return all_images

    
    
    
def LoadImagenet21kVal(num_classes=1500, num_per_class=20, separate_classes=False):
    
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
    
    all_images = []
    for image in sorted(os.listdir(path)):
        all_images.append(f'{path}/{image}')
    return all_images    
    

    
    
    
def LoadImagePaths(name, mode, *args, **kwargs): 
    
    if name == 'object2vec':
        return LoadObject2VecImages()
        
    elif name == 'imagenet21k':
        return LoadImagenet21kImages(*args, **kwargs)

    elif name == 'imagenet21k_val':
        return LoadImagenet21kVal(*args, **kwargs)
    
    elif 'naturalscenes' in name:
        
        if mode == 'cv':
            return LoadNSDImages(shared_images=True)
        
        elif mode == 'pca':
            return LoadNSDImages(subset=True)
        
        else: 
            return LoadNSDImages()
    
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
    
    
    if 'imagenet' in dataset:
        return [os.path.basename(i).strip('.JPEG') for i in images]
