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
from config import CACHE 

IDS_PATH = os.path.join(ROOT, 'image_tools','nsd_ids_unshared_sample=30000')
NSD_UNSHARED_SAMPLE = [image_id.strip('.png') for image_id in pickle.load(open(IDS_PATH, 'rb'))]


DATASET = 'naturalscenes'
DEVICE = 'cuda'
models = ['expansion_10000']
MODE = 'pca'



for model_name in models:
        
    print('computing PCs')
    model_info = load_model_dict(model_name)
    
    activations_identifier = get_activations_iden(model_info=model_info, dataset= DATASET)
    
    Activations(model=model_info['model'],
                    layer_names=model_info['layers'],
                    dataset=DATASET,
                    device= DEVICE,
                    batch_size = 80).get_array(activations_identifier) 

    data = xr.open_dataarray(os.path.join(CACHE,'activations',activations_identifier),engine='netcdf4')
    
    activations_identifier = activations_identifier + '_principal_components'
    
    if DATASET == 'naturalscenes':
        data = filter_activations(data, NSD_UNSHARED_SAMPLE)
        _PCA()._fit(activations_identifier, data)
    
    else:
        _PCA()._fit(activations_identifier, data.values)



