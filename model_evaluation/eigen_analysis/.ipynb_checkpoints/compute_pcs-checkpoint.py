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
from model_features.models.models import load_model, load_iden
from model_evaluation.predicting_brain_data.benchmarks.nsd import filter_activations   
from model_evaluation.predicting_brain_data.benchmarks.majajhong import load_activations  
from image_tools.loading import get_image_labels
from config import CACHE 
IDS_PATH = os.path.join(ROOT, 'image_tools','nsd_ids_unshared_sample=30000')
NSD_UNSHARED_SAMPLE = [image_id.strip('.png') for image_id in pickle.load(open(IDS_PATH, 'rb'))]

MODEL_NAME = 'expansion'
FEATURES = 3000
DATASET = 'naturalscenes'
DEVICE = 'cuda'
COMPONENTS = 10000 
print('computing PCs')

        
activations_identifier = load_iden(model_name=MODEL_NAME, features=FEATURES, layers=5, dataset=DATASET)
print(activations_identifier)
model = load_model(model_name=MODEL_NAME, features=FEATURES, layers=5)

Activations(model=model,
            layer_names=['last'],
            dataset=DATASET,
            device= DEVICE,
            batch_size = 10000,
            compute_mode = 'fast').get_array(activations_identifier)   


data = xr.open_dataarray(os.path.join(CACHE,'activations',activations_identifier),engine='netcdf4')

        
iden = activations_identifier + f'_components={COMPONENTS}'
print(iden)

if DATASET == 'naturalscenes':
    data = filter_activations(data, NSD_UNSHARED_SAMPLE)
    _PCA(n_components = COMPONENTS)._fit(iden, data)

elif DATASET == 'majajhong':
    data = load_activations(activations_identifier, mode = 'train')
    _PCA(n_components = COMPONENTS)._fit(iden, data)

else: 
    data = data.values
    _PCA(n_components = COMPONENTS)._fit(iden, data)

