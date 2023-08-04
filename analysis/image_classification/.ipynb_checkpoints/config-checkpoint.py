import pickle
import os
import sys
from tools.loading import *


CAT_NAMES = load_places_cat_names()

with open('/data/atlas/datasets/places/categories_subset_100','rb') as f:
    CAT_SUBSET = pickle.load(f)

CAT_LABELS = load_places_cat_labels()
CAT_LABELS_SUBSET = {k: v for k, v in CAT_LABELS.items() if v in CAT_SUBSET}
VAL_IMAGES_SUBSET = list(CAT_LABELS_SUBSET.keys())

