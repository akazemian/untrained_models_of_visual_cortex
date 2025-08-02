import pickle
from image_tools.loading import load_places_cat_ids, load_places_cat_labels
from config import PLACES_IMAGES
import random
random.seed(42)


CAT_LABELS = load_places_cat_labels()
VAL_IMAGES = list(CAT_LABELS.keys())
CAT_NAMES = load_places_cat_ids()
    
