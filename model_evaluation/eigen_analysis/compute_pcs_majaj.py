import os
import sys
import pickle
ROOT = os.getenv('BONNER_ROOT_PATH')
sys.path.append(ROOT)
import warnings
warnings.filterwarnings('ignore')
import xarray as xr
from model_evaluation.utils import get_activations_iden
from model_features.activation_extractor import Activations
from model_evaluation.eigen_analysis.utils import _PCA
from model_features.models.models import load_model_dict
from model_evaluation.predicting_brain_data.benchmarks.nsd import filter_activations   
from model_evaluation.predicting_brain_data.benchmarks.majajhong import load_activations  
from image_tools.loading import get_image_labels
from config import CACHE 

DATASET = 'majajhong'
DEVICE = 'cuda'

MODE = 'pca'
GLOBAL_POOL = False 

        
print('computing PCs')

#'expansion_5_layers_100000_6x6_gmp'
# Activations(model=Expansion5L(filters_5=30000, gpool = False).Build() ,
#             layer_names=['last'],
#             dataset=DATASET,
#             device= DEVICE,
#             batch_size = 2,
#             compute_mode = 'fast').get_array(activations_identifier)   

data = xr.open_dataarray(os.path.join(CACHE,'activations','expansion_5_layers_30000_6x6'),engine='netcdf4')

activations_identifier = 'expansion_5_layers_30000_6x6_principal_components'

if DATASET == 'naturalscenes':
    data = filter_activations(data, NSD_UNSHARED_SAMPLE)
    _PCA()._fit(activations_identifier, data)

else:
    data = load_activations('expansion_5_layers_30000_6x6', mode = 'train')
    _PCA()._fit(activations_identifier, data)



