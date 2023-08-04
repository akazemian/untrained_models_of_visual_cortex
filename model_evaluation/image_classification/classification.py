import warnings
warnings.filterwarnings('ignore')

# libraries
import sys
import os
from tqdm import tqdm
import pickle
import xarray as xr

# local libraries
from config import VAL_IMAGES_SUBSET, RESULTS_PATH
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
from data_tools.config import ACTIVATIONS_PATH
from model_evaluation.image_classification.tools import get_pairwise_performance, normalize
from model_evaluation.utils import get_activations_iden
from model_features.activation_extractor import Activations

# models
from model_features.models.expansion_3_layers import ExpansionModel
from model_features.models.alexnet import Alexnet



# local vars
DATASET = 'places'
HOOK = None



# define models in a dict
model_dict = {
    
    'expansion':{
                'iden':'expansion_model_test',
                'model':ExpansionModel(filters_3=10000).Build(),
                'layers': ['last'], 
                'num_layers':3,
                'num_features':10000,
    },
    
    'alexnet':{  
                'iden':'alexnet_conv5',
                'model':Alexnet().Build(),
                'layers': ['last'], 
                'num_layers':5,
                'num_features':256,
    }
}


for model_info in model_dict.values():
    
    activations_iden = get_activations_iden(model_info=model_info, dataset=DATASET, mode=None)
    
    activations = Activations(model=model_info['model'],
                            layer_names=model_info['layers'],
                            dataset=DATASET,
                            mode = None,
                            hook = HOOK,
                            batch_size = 50)


    # extract model activations
    activations.get_array(ACTIVATIONS_PATH,activations_iden) 
    data = xr.open_dataset(os.path.join(ACTIVATIONS_PATH,activations_iden))
    
    # normalize activations for image classification
    data.x.values = normalize(data.x.values)
    
    # take the subset of activations belonging to the 100 categories of images
    data = data.set_xindex('stimulus_id')
    data_subset = data.sel(stimulus_id = VAL_IMAGES_SUBSET)


    # get pairwise classification performance
    performance_dict = get_pairwise_performance(data_subset)

    with open(os.path.join(RESULTS_PATH,'classification',activations_iden),'wb') as f:
        pickle.dump(performance_dict,f)

    print(f'pairwaise performance is saved in {os.path.join(RESULTS_PATH,"classification",activations_iden)}')