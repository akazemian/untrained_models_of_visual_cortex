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
from image_tools.loading import load_places_cat_labels
from model_evaluation.image_classification.tools import get_Xy, cv_performance
from model_features.activation_extractor import Activations
from config import CACHE
from model_features.models.models import load_model, load_iden


# local vars
DATASET = 'places_val'
HOOK = 'pca'
features = 3000
layers = 5

# define models in a dict
model_name = 'expansion'
model_name = '_alexnet'


activations_identifier = load_iden(model_name=model_name, dataset=DATASET, features=features, layers=layers) + f'_principal_components=1000'
pca_iden = load_iden(model_name=model_name, dataset='places_train', features=features, layers=layers)  + f'_components=1000'

model = load_model(model_name=model_name, features = features, layers=layers)


Activations(model=model,
        layer_names=['last'],
        dataset=DATASET,
        device='cuda',
        hook=HOOK,
        pca_iden = pca_iden,
        batch_size=100).get_array(activations_identifier)

data = xr.open_dataset(os.path.join(CACHE,'activations',activations_identifier))
data = data.set_xindex('stimulus_id')

cat_labels = load_places_cat_labels()
X, y = get_Xy(data)
score = cv_performance(X, y, cat_labels)
print(activations_identifier, ':', score)

with open(os.path.join(CACHE,'classification',activations_identifier+'_stratified'),'wb') as f:
    pickle.dump(score,f)



