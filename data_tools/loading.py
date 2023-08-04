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
sys.path.append(os.getenv('MB_ROOT_PATH'))
from data_tools.config import PLACES_PATH #,NSD_PATH, NSD_SHARED_IDS, NSD_SUBSET_IDS, MAJAJHONG_PATH, 
    

# def load_nsd_images(shared_images=False,unshared_images=False,subset=False):
     
#     print('shared images:',shared_images)
#     print('unshared images:',unshared_images)
#     print('subset images:',subset)


#     all_images = []
           
#     if subset: 
#         #file_name = 
#         return pickle.load(open(os.path.join(PATH_TO_NSD_SUBSET_IDS,file_name),'rb'))

    
#     else:
        
#         shared_ids = pickle.load(open(PATH_TO_NSD_SHARED_IDS, 'rb'))   
        
#         for image in sorted(os.listdir(NSD_PATH)):
                
#             if shared_images:
#                 if image.strip('.png') in shared_ids:
#                     all_images.append(f'{NSD_PATH}/{image}')
            
#             elif unshared_images:
#                 if image.strip('.png') not in shared_ids:
#                     all_images.append(f'{NSD_PATH}/{image}')            
            
#             else:
#                 all_images.append(f'{NSD_PATH}/{image}')

#     return all_images 
    

        
# def load_majajhong_images():
    
#     all_images = []
#     for image in sorted(os.listdir(MAJAJHONG_PATH)):
#         all_images.append(f'{MAJAJHONG_PATH}/{image}')
#     return all_images

    
    
def load_places_images():

        val_images = os.listdir(os.path.join(PLACES_PATH,'val_images/val_256'))
        val_images_paths = [f'{PLACES_PATH}/val_images/val_256/{i}' for i in val_images]
        
        return sorted(val_images_paths)
    
    
    
def load_image_paths(name, mode, *args, **kwargs): 
    
    match name:
        
        case 'naturalscenes':
            
            if mode == 'cv':
                return load_nsd_images(shared_images=True)
            
            elif mode == 'pca':
                return load_nsd_images(subset=True)
            
            return load_nsd_images()

        case 'majajhong':
            return load_majajhong_images()

        case 'places':
            return load_places_images()
    

    
    
def get_image_labels(dataset,images,*args,**kwargs):
    
    match dataset:
        
        case 'majajhong':
            name_dict = pd.read_csv('/data/shared/brainio/brain-score/image_dicarlo_hvm-public.csv').set_index('image_file_name')['image_id'].to_dict()
            return [name_dict[os.path.basename(i)] for i in images]

        case 'naturalscenes':
            return [os.path.basename(i).strip('.png') for i in images]

        case 'places':
            return [os.path.basename(i) for i in images]
                                                  
    
    
def load_places_cat_labels():
    
    with open(os.path.join(PLACES_PATH,'places365_val.txt'), "r") as file:
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
    
    val_image_paths = load_places_images()
    val_image_names = [os.path.basename(i) for i in val_image_paths]
    cat_dict = load_places_cat_labels()

    return [cat_dict[i] for i in val_image_names]              
            


