import os
import sys
import pickle
ROOT = os.getenv('BONNER_ROOT_PATH')
sys.path.append(ROOT)
import warnings
warnings.filterwarnings('ignore')
import xarray as xr
from model_features.activation_extractor import Activations
from model_evaluation.eigen_analysis.utils import _PCA
from model_evaluation.predicting_brain_data.benchmarks.nsd import filter_activations    
from model_evaluation.predicting_brain_data.benchmarks.majajhong import load_activations    
from model_features.models.expansion_5_layers import Expansion5L
from config import CACHE 
from model_features.models.models import load_model, load_iden

IDS_PATH = os.path.join(ROOT, 'image_tools','nsd_ids_unshared_sample=30000')
NSD_UNSHARED_SAMPLE = [image_id.strip('.png') for image_id in pickle.load(open(IDS_PATH, 'rb'))]


DATASET = 'majajhong'
DEVICE = 'cuda'
GLOBAL_POOL = False 


# activations_identifier = load_iden(model_name='expansion', features=3000, dataset=DATASET)
# model = load_model(model_name='expansion', features=3000, layers=None)

activations_identifier = load_iden(model_name='alexnet', features=None, layers=5, dataset=DATASET)
model = load_model(model_name='alexnet', features=None, layers=5)

print(activations_identifier)

Activations(model=model,
                layer_names='last',
                dataset=DATASET,
                device= DEVICE,
                batch_size = 40).get_array(activations_identifier) 

data = xr.open_dataarray(os.path.join(CACHE,'activations',activations_identifier),engine='netcdf4')

pca_identifier = activations_identifier + '_principal_components'

if DATASET == 'naturalscenes':
    data = filter_activations(data, NSD_UNSHARED_SAMPLE)
    _PCA()._fit(pca_identifier, data)

elif DATASET == 'majajhong':
    data = load_activations(activations_identifier=activations_identifier, mode='train')
    _PCA()._fit(pca_identifier, data)

else:
    print('dataset does not exist')

