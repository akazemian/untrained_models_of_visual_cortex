import warnings
warnings.filterwarnings('ignore')

# libraries
import sys
import os
from tqdm import tqdm
import pickle
import xarray as xr

# local libraries
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
from model_evaluation.image_classification._config import VAL_IMAGES_SUBSET
from model_evaluation.image_classification.tools import PairwiseClassification, normalize
from model_evaluation.utils import get_activations_iden
from model_features.activation_extractor import Activations
from config import CACHE

# models
from model_features.models.models import load_model_dict


# local vars
DATASET = 'places'
HOOK = None



# define models in a dict
models = ['expansion_10000'] #, 'alexnet_conv5']

for model_name in models:
    
    print(model_name)
    model_info = load_model_dict(model_name)

    activations_iden = get_activations_iden(model_info=model_info, dataset=DATASET)
    
    activations = Activations(model=model_info['model'],
                            layer_names=model_info['layers'],
                            dataset=DATASET,
                            hook = HOOK,
                            batch_size = 50,
                            compute_mode = 'slow').get_array(activations_iden) 
    
    data = xr.open_dataset(os.path.join(CACHE,'activations',activations_iden))
    
    # normalize activations for image classification
    data.x.values = normalize(data.x.values)
    
    # take the subset of activations belonging to the 100 categories of images
    data = data.set_xindex('stimulus_id')
    data_subset = data.sel(stimulus_id = VAL_IMAGES_SUBSET)

    # get pairwise classification performance
    PairwiseClassification().get_performance(iden = activations_iden, 
                                            data = data_subset)
