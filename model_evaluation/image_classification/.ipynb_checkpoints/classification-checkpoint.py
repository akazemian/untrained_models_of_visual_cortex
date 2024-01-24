import warnings
warnings.filterwarnings('ignore')

# libraries
import sys
import os
from tqdm import tqdm
import pickle
import xarray as xr
import numpy as np
# local libraries
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
from model_evaluation.image_classification._config import VAL_IMAGES_SUBSET
from model_evaluation.image_classification.tools import PairwiseClassification, normalize
from model_features.activation_extractor import Activations
from config import CACHE
from model_features.models.models import load_model, load_iden


# models


# local vars
DATASET = 'places'
HOOK = None
features = None
layers = 5

# define models in a dict
#model_name = 'expansion'
model_name = 'alexnet'

activations_identifier = load_iden(model_name=model_name, dataset=DATASET, features=features, layers=layers)

model = load_model(model_name=model_name, features = features, layers=layers)

Activations(model=model,
        layer_names=['last'],
        dataset=DATASET,
        device='cuda',
        batch_size=10).get_array(activations_identifier)

data = xr.open_dataset(os.path.join(CACHE,'activations',activations_identifier))

# normalize activations for image classification
data.x.values = normalize(data.x.values)
# replace NaNs with zero
data.x.values = np.nan_to_num(data.x.values)
# take the subset of activations belonging to the 100 categories of images

data = data.set_xindex('stimulus_id')
data_subset = data.sel(stimulus_id = VAL_IMAGES_SUBSET)

# get pairwise classification performance
PairwiseClassification().get_performance(iden = activations_identifier + '_normalized', 
                                        data = data_subset)
