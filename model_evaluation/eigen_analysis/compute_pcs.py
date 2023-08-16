import os
import sys
import pickle
ROOT = os.getenv('BONNER_ROOT_PATH')
sys.path.append(ROOT)
import warnings
warnings.filterwarnings('ignore')
import xarray as xr
import torchvision
from model_evaluation.predicting_brain_data.regression.scorer import EncodingScore
from model_evaluation.utils import get_activations_iden, get_scores_iden
from model_features.activation_extractor import Activations
from model_evaluation.eigen_analysis.utils import _PCA
from model_features.models.models import load_model_dict
from model_evaluation.predicting_brain_data.benchmarks.nsd import filter_activations    
from config import CACHE 

IDS_PATH = os.path.join(ROOT, 'image_tools','nsd_ids_unshared_sample=30000')
NSD_UNSHARED_SAMPLE = [image_id.strip('.png') for image_id in pickle.load(open(IDS_PATH, 'rb'))]


DATASET = 'majajhong'
DEVICE = 'cuda'
models = ['expansion_10000','alexnet_conv5']
MODE = 'pca'



for model_name in models:
        
    print(model_name)
    model_info = load_model_dict(model_name)
    
    activations_identifier = get_activations_iden(model_info=model_info, dataset= DATASET)
    
    Activations(model=model_info['model'],
                    layer_names=model_info['layers'],
                    dataset=DATASET,
                    device= DEVICE,
                    batch_size = 80).get_array(activations_identifier) 

    X = xr.open_dataarray(os.path.join(CACHE,'activations',activations_identifier),engine='netcdf4').values
    
    if DATASET == 'naturalscenes':
        X = filter_activations(X, NSD_UNSHARED_SAMPLE)

    _PCA()._fit(activations_identifier, X)
    



