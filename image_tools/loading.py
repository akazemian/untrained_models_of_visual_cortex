import os
import pandas as pd
import sys
ROOT = os.getenv('BONNER_ROOT_PATH')
sys.path.append(ROOT)
from config import PLACES_IMAGES, NSD_IMAGES, MAJAJ_IMAGES, MAJAJ_NAME_DICT 


def load_nsd_images():
    """
    Loads the file paths of natural scene images from the NSD_IMAGES directory.

    Returns:
        list: A sorted list of full paths to the natural scene images.
    """
    return sorted([os.path.join(NSD_IMAGES,image) for image in os.listdir(NSD_IMAGES)])
    
        
def load_majaj_images():
    """
    Loads the file paths of images from the MAJAJ_IMAGES directory.

    Returns:
        list: A sorted list of full paths to the images in the MAJAJ_IMAGES directory.
    """
    return sorted([f'{MAJAJ_IMAGES}/{image}' for image in os.listdir(MAJAJ_IMAGES)])
    
    
def load_places_images():
    """
    Loads the file paths of validation images from the PLACES_IMAGES directory.

    Returns:
        list: A sorted list of full paths to the validation images.
    """
    val_images = os.listdir(os.path.join(PLACES_IMAGES,'val_images/val_256'))
    val_images_paths = [f'{PLACES_IMAGES}/val_images/val_256/{i}' for i in val_images]
    
    return sorted(val_images_paths)
    
    
def load_image_paths(name): 
    
    """
    Load image file paths based on a specified dataset name.

    Args:
        name (str): Name of the dataset ('naturalscenes', 'majajhong', or 'places').

    Returns:
        list: A sorted list of full paths to the images for the specified dataset.
    """
    
    match name:
        
        case 'naturalscenes':
            return load_nsd_images()

        case 'majajhong':
            return load_majaj_images()

        case 'places':
            return load_places_images()
    
  
    

def get_image_labels(dataset, images):
    
    """
    Get image labels based on a specified dataset.

    Args:
        dataset (str): Name of the dataset ('naturalscenes', 'majajhong', or 'places').
        images (list): List of image file paths for which to obtain labels.

    Returns:
        list: List of labels corresponding to the provided images.
    """
    
    match dataset:
        
        case 'naturalscenes':
            return [os.path.basename(i).strip('.png') for i in images]

        
        case 'majajhong':
            name_dict = pd.read_csv(MAJAJ_NAME_DICT).set_index('image_file_name')['image_id'].to_dict()
            return [name_dict[os.path.basename(i)] for i in images]


        case 'places':
            return [os.path.basename(i) for i in images]
                                                  
    

def load_places_cat_labels():
    """
    Load category labels for placees dataset.

    Returns:
        dict: Dictionary where keys are image filenames and values are category labels.
    """    
    with open(os.path.join(PLACES_IMAGES,'places365_val.txt'), "r") as file:
        content = file.read()
    annotations = content.split('\n')
    cat_dict = {}
    for annotation in annotations:
        image = annotation.split(' ')[0]
        cat = annotation.split(' ')[1]
        cat_dict[image] = int(cat)
    return cat_dict
    
   
    
def load_places_cat_names():
    """
    Load category names for the places dataset.

    Returns:
        list: List of category names for the validation images in the PLACES_IMAGES dataset.
    """    
    val_image_paths = load_places_images()
    val_image_names = [os.path.basename(i) for i in val_image_paths]
    cat_dict = load_places_cat_labels()

    return [cat_dict[i] for i in val_image_names]              
            


