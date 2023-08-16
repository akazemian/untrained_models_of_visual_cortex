import os
import pandas as pd
import sys
import pickle
ROOT = os.getenv('BONNER_ROOT_PATH')
sys.path.append(ROOT)
from config import PLACES_IMAGES, NSD_IMAGES, MAJAJ_IMAGES, MAJAJ_NAME_DICT 


def load_nsd_images():
    return sorted([os.path.join(NSD_IMAGES,image) for image in os.listdir(NSD_IMAGES)])
    

        
def load_majaj_images():
    return sorted([f'{MAJAJ_IMAGES}/{image}' for image in os.listdir(MAJAJ_IMAGES)])

    
    
def load_places_images():

        val_images = os.listdir(os.path.join(PLACES_IMAGES,'val_images/val_256'))
        val_images_paths = [f'{PLACES_IMAGES}/val_images/val_256/{i}' for i in val_images]
        
        return sorted(val_images_paths)
    
    
    
def load_image_paths(name, *args, **kwargs): 
    
    match name:
        
        case 'naturalscenes':
            return load_nsd_images()

        case 'majajhong':
            return load_majaj_images()

        case 'places':
            return load_places_images()
    

    
    
def get_image_labels(dataset,images,*args,**kwargs):
    
    match dataset:
        
        case 'naturalscenes':
            return [os.path.basename(i).strip('.png') for i in images]

        
        case 'majajhong':
            name_dict = pd.read_csv(MAJAJ_NAME_DICT).set_index('image_file_name')['image_id'].to_dict()
            return [name_dict[os.path.basename(i)] for i in images]


        case 'places':
            return [os.path.basename(i) for i in images]
                                                  
    
    

def load_places_cat_labels():
    
    with open(os.path.join(PLACES_IMAGES,'places365_val.txt'), "r") as file:
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
            


