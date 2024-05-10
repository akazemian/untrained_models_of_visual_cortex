import os
import pandas as pd
import sys
import pickle
ROOT = os.getenv('BONNER_ROOT_PATH')
sys.path.append(ROOT)


def load_nsd_images(ids=None):
    from config import NSD_IMAGES
    """
    Loads the file paths of natural scene images from the NSD_IMAGES directory.

    Returns:
        list: A sorted list of full paths to the natural scene images.
    """
    if ids is not None:        
        return sorted([f'{os.path.join(NSD_IMAGES,image)}.png' for image in ids])
        
    else:
        return sorted([os.path.join(NSD_IMAGES,image) for image in os.listdir(NSD_IMAGES)])
    
    
        
def load_majaj_images():
    from config import MAJAJ_IMAGES
    """
    Loads the file paths of images from the MAJAJ_IMAGES directory.

    Returns:
        list: A sorted list of full paths to the images in the MAJAJ_IMAGES directory.
    """
    return sorted([f'{MAJAJ_IMAGES}/{image}' for image in os.listdir(MAJAJ_IMAGES)])
    
    
    
    
def load_places_val_images():
    from config import PLACES_IMAGES
    """
    Loads the file paths of validation images from the PLACES_IMAGES directory.

    Returns:
        list: A sorted list of full paths to the validation images.
    """
        
    images = os.listdir(os.path.join(PLACES_IMAGES,'val_images/val_256'))
    images_paths = [f'{PLACES_IMAGES}/val_images/val_256/{i}' for i in images]
    
    return sorted(images_paths)
    
    
def load_places_train_images():
    from config import PLACES_IMAGES
    
    images_paths = []
    base_dir = os.path.join(PLACES_IMAGES,'train_images_subset')
    
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for subdir in subdirs:
        # List all files in the subdirectory, including their full paths
        images_paths.extend([os.path.join(subdir, f) for f in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, f))])

    return base_dir, sorted(images_paths)

    

    
    

def load_image_paths(name, *args, **kwargs): 
    
    """
    Load image file paths based on a specified dataset name.

    Args:
        name (str): Name of the dataset ('naturalscenes', 'majajhong', or 'places').

    Returns:
        list: A sorted list of full paths to the images for the specified dataset.
    """
    
    match name:
        
        case 'naturalscenes' | 'naturalscenes_shuffled':
            return load_nsd_images(*args, **kwargs)
            
        case 'majajhong' | 'majajhong_shuffled':
            return load_majaj_images()

        case 'places_val':
            return load_places_val_images()
    
        case 'places_train':
            return load_places_train_images()



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
        
        case 'naturalscenes' | 'naturalscenes_shuffled':
            return [int(os.path.basename(i).strip('image.png')) for i in images]
        
        case 'majajhong' | 'majajhong_shuffled':
            from config import MAJAJ_NAME_DICT 
            name_dict = pd.read_csv(MAJAJ_NAME_DICT).set_index('image_file_name')['image_id'].to_dict()
            return [name_dict[os.path.basename(i)] for i in images]
        
        case 'places_train':
            return [multi_level_basename(i) for i in images]
                                                  
    

def load_places_cat_labels():
    from config import PLACES_IMAGES
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
    
   
 
    
    
def load_places_cat_ids():
    """
    Load category names for the places dataset.

    Returns:
        list: List of category names for the validation images in the PLACES_IMAGES dataset.
    """    
    val_image_paths = load_places_val_images()
    val_image_names = [os.path.basename(i) for i in val_image_paths]
    cat_dict = load_places_cat_labels()

    return [cat_dict[i] for i in val_image_names]              
            


def multi_level_basename(full_path, levels=2):
    # Normalize the path first
    full_path = os.path.normpath(full_path)

    # Break the path into parts
    path_parts = full_path.split(os.sep)

    # Get the last 'levels' parts of the path
    if len(path_parts) >= levels:
        result = os.path.join(*path_parts[-levels:])
    else:
        result = os.path.join(*path_parts)

    return result


