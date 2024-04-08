import pickle
import os
import sys
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
from image_tools.loading import load_places_cat_ids, load_places_cat_labels
from config import PLACES_IMAGES
import random
random.seed(42)


CAT_SUBSET_PATH = os.path.join(PLACES_IMAGES,'categories_subset_100')
if not os.path.exists(PLACES_IMAGES):
    print('generating a subset of 100 categories')
    with open(os.path.join(PLACES_IMAGES,'categories_places365.txt'),'r') as f:
        categories = f.read().split('\n/') 
    
    CAT_SUBSET = random.sample(range(0, len(num_categories)), 100)
    with open(CAT_SUBSET_PATH,'wb') as f:
        pickle.dump(CAT_SUBSET,f)
                            
else:
    with open(CAT_SUBSET_PATH,'rb') as f:
        CAT_SUBSET = pickle.load(f)


CAT_LABELS = load_places_cat_labels()
CAT_LABELS_SUBSET = {k: v for k, v in CAT_LABELS.items() if v in CAT_SUBSET}
VAL_IMAGES_SUBSET = list(CAT_LABELS_SUBSET.keys())
CAT_NAMES = load_places_cat_ids()
    
