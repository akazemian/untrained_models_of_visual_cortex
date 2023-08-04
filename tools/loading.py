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
import sys
ROOT_DIR = os.getenv('MB_ROOT_PATH')
sys.path.append(ROOT_DIR)
from tools.utils import *



ROOT = os.getenv('MB_DATA_PATH')
ROOT_SHARED = os.getenv('MB_SHARED_DATA_PATH')


PATH_TO_NSD_SHARED_IDS = os.path.join(ROOT,'neural_data/nsd_shared_ids')
PATH_TO_NSD_SUBSET_IDS = os.path.join(ROOT,'neural_data')
NSD_PATH = os.path.join(ROOT_SHARED,'datasets/allen2021.natural_scenes/images')
MAJAJHONG_PATH = os.path.join(ROOT_SHARED,'brainio/brain-score/dicarlo.hvm-public')
IMAGENET_21K_PATH = os.path.join(ROOT_SHARED,'datasets/ilsvrc2012/train')
PATH_TO_PLACES = os.path.join(ROOT,'datasets/places')
NUM_CLASSES = 100
NUM_PER_CLASS = 50
    





def LoadObject2VecImages():
    
    all_images = []
    for folder in os.listdir(OBJECT_2_VEC_PATH):
        for image in os.listdir(f'{data_path}/{folder}'):
            all_images.append(f'{data_path}/{folder}/{image}')
    return all_images    
    
    
    
    
def LoadPlacesImages(subset=False):
                    
    if subset: 
        
        file_name = 'test_images_ids_subset'
        
        if not os.path.exists(os.path.join(PATH_TO_PLACES,file_name)):
            test_images = os.listdir(os.path.join(PATH_TO_PLACES,'test_images/test_256'))
            test_images_subset = random.sample(test_images,10000)
            test_images_subset_paths = [f'{PATH_TO_PLACES}/test_images/test_256/{i}' for i in test_images_subset]
            with open(os.path.join(PATH_TO_PLACES,file_name),'wb') as f:
                pickle.dump(test_images_subset_paths,f)
        
        test_images_subset_paths = pickle.load(open(os.path.join(PATH_TO_PLACES,file_name),'rb'))
        return sorted(test_images_subset_paths)
    
    
    else:
        val_images = os.listdir(os.path.join(PATH_TO_PLACES,'val_images/val_256'))
        val_images_paths = [f'{PATH_TO_PLACES}/val_images/val_256/{i}' for i in val_images]
        return sorted(val_images_paths)

    


def LoadNSDImages(shared_images=False,unshared_images=False,subset=False):
     
    print('shared images:',shared_images)
    print('unshared images:',unshared_images)
    print('subset images:',subset)


    num_samples=30000
    all_images = []
           
    if subset: 
        file_name = f'nsd_sample_ids_{num_samples}'
        if not os.path.exists(os.path.join(PATH_TO_NSD_SUBSET_IDS,file_name)):
            generate_nsd_sample(num_samples=num_samples,file_name=file_name)
        return pickle.load(open(os.path.join(PATH_TO_NSD_SUBSET_IDS,file_name),'rb'))

    
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

    
    
    
def LoadImagenet21kImages(num_classes=1000,num_per_class=10):
    
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


    
def LoadImagenet21kVal(num_classes=NUM_CLASSES, num_per_class=NUM_PER_CLASS):
    
    base_indices = np.arange(num_per_class).astype(int)
    indices = []
    for i in range(num_classes):
        indices.extend(50 * i + base_indices)

    framework_home = os.path.expanduser(os.getenv('MT_HOME', '~/model-tools'))
    imagenet_filepath = os.getenv('MT_IMAGENET_PATH', os.path.join(framework_home, 'imagenet2012.hdf5'))
    imagenet_dir = f"{imagenet_filepath}-files"
    os.makedirs(imagenet_dir, exist_ok=True)

    if not os.path.isfile(imagenet_filepath):
        os.makedirs(os.path.dirname(imagenet_filepath), exist_ok=True)
        download_file("imagenet2012-val.hdf5", imagenet_filepath)

    filepaths = []
    with h5py.File(imagenet_filepath, 'r') as f:
        for index in indices:
            imagepath = os.path.join(imagenet_dir, f"{index}.png")
            if not os.path.isfile(imagepath):
                image = np.array(f['val/images'][index])
                Image.fromarray(image).save(imagepath)
            filepaths.append(imagepath)

    # if separate_classes:
    #     filepaths = [filepaths[i * num_per_class:(i + 1) * num_per_class]
    #                 for i in range(num_classes)]
        
    return filepaths




def LoadLocalizerImages():
    
    all_images = []
    for image in sorted(os.listdir(path)):
        all_images.append(f'{path}/{image}')
    return all_images  
                                                

    
    
    
def LoadImagePaths(name, mode, *args, **kwargs): 
    
    match name:
        
        case 'object2vec':
            return LoadObject2VecImages()

        case 'imagenet21k':
            return LoadImagenet21kImages(*args, **kwargs)

        case 'imagenet21kval':
            return LoadImagenet21kVal(*args, **kwargs)

        case 'naturalscenes':
            
            if mode == 'cv':
                return LoadNSDImages(shared_images=True)
            
            elif mode == 'pca':
                return LoadNSDImages(subset=True)
            
            return LoadNSDImages()

        case 'majajhong':
            return LoadMajajHongImages()

        case 'localizers':
            return LoadLocalizerImages()

        case 'places':
            
            if mode == 'pca':
                return LoadPlacesImages(subset=True)
            
            return LoadPlacesImages()
    

    
    
def get_image_labels(dataset,images,*args,**kwargs):
    
    match dataset:
        
        case 'majajhong':
            name_dict = pd.read_csv('/data/shared/brainio/brain-score/image_dicarlo_hvm-public.csv').set_index('image_file_name')['image_id'].to_dict()
            return [name_dict[os.path.basename(i)] for i in images]


        case 'naturalscenes':
            return [os.path.basename(i).strip('.png') for i in images]


        case 'imagenet21k':
            return [os.path.basename(i).strip('.JPEG') for i in images]

        case 'imagenet21kval':
            return [j for j in range(NUM_CLASSES) for i in range(NUM_PER_CLASS)]
        
        case 'places':
            return [os.path.basename(i) for i in images]


        
                                                  
            
            
def load_places_cat_labels():
    
    with open(os.path.join(PATH_TO_PLACES,'places365_val.txt'), "r") as file:
        content = file.read()
    annotations = content.split('\n')
    cat_dict = {}
    cats = []
    for annotation in annotations:
        image = annotation.split(' ')[0]
        cat = annotation.split(' ')[1]
        cat_dict[image] = int(cat)
    return cat_dict
    
    
    
    
    
def load_places_cat_names():
    
    val_image_paths = LoadPlacesImages()
    val_image_names = [os.path.basename(i) for i in val_image_paths]
    cat_dict = load_places_cat_labels()

    return [cat_dict[i] for i in val_image_names]  




