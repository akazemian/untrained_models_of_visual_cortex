    
import os 
import sys
ROOT = os.getenv('BONNER_ROOT_PATH')
sys.path.append(ROOT)

import warnings
warnings.filterwarnings('ignore')
import xarray as xr
import torchvision
from image_tools.processing import *
from model_evaluation.predicting_brain_data.regression.scorer import EncodingScore
from model_evaluation.utils import get_activations_iden, get_scores_iden
from model_features.activation_extractor import Activations
from model_features.models.expansion_3_layers import Expansion
from model_evaluation.eigen_aalysis.utils import _PCA
    
    
DATASET = 'naturalscenes'
N_COMPONENTS = 256
DEVICE = 'cuda'
 
    
    
model_info = {
                'iden':'expansion_model',
                'model': Expansion(filters_3=10000).Build(),
                'layers': ['last'], 
                'num_layers': 3,
                'num_features': 10000
    }
        

       
    activations_identifier = get_activations_iden(model_info, DATASET, MODE)
    
    Activations(model=model_info['model'],
                layer_names=model_info['layers'],
                dataset=DATASET,
                mode = 'pca',
                hook = None,
                device= DEVICE,
                batch_size = 80).get_array(activations_identifier) 
    
    _PCA(n_components=N_COMPONENTS).fit(activations_identifier)
    



