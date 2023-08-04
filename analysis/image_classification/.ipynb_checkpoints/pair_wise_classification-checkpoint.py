# model wrapper
import sys
import os
ROOT_DIR = os.getenv('MB_ROOT_PATH')
sys.path.append(ROOT_DIR)
from models.all_models.model_3L_abs_blurpool_avgpool import ExpansionModel
from models.all_models.alexnet import Alexnet
from tqdm import tqdm
import pickle
from analysis.image_classification.train import *
from analysis.image_classification import config
from tools.processing import *
import xarray as xr
from tools.processing import *


DATA_DIR = os.getenv('MB_DATA_PATH')
ACTIVATIONS_PATH = os.path.join(DATA_DIR,'activations') 
DATASET, MODE = 'places', None
HOOK = None
MAX_POOL = True



model_info = {
                'iden':'expansion_model_final_original_features_relu',
                'model':ExpansionModel(filters_3=10000).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                'num_layers':3,
                'num_features':10000,
                'max_pool':MAX_POOL,
}
# model_info = {   
#                 'iden':'alexnet_conv5',
#                 'model':Alexnet().Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#                 'num_layers':5,
#                 'num_features':256,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
# }

activations_identifier = get_activations_iden(model_info, DATASET, MODE)

activations = Activations(model=model_info['model'],
                        layer_names=model_info['layers'],
                        dataset=DATASET,
                        preprocess=model_info['preprocess'],
                        mode = MODE,
                        _hook = HOOK,
                        batch_size = 50)


activations.get_array(ACTIVATIONS_PATH,activations_identifier) 
data = xr.open_dataset(os.path.join(ACTIVATIONS_PATH,activations_identifier))
data.x.values = normalize(data.x.values)
data_subset = data.where(data.stimulus_id.isin(config.VAL_IMAGES_SUBSET), drop=True)
data_subset = data_subset.set_xindex('stimulus_id')


performance_dict = {}
pairs = []

for cat_1 in tqdm(config.CAT_SUBSET):
    for cat_2 in config.CAT_SUBSET:
        
        if {cat_1, cat_2} in pairs:
            pass
            
        elif cat_1 == cat_2:
            performance_dict[(cat_1,cat_2)] = 1

        else:
            X, y = get_Xy(data_subset, [cat_1,cat_2])
            performance_dict[(cat_1,cat_2)] = cv_performance(X, y)
            pairs.append({cat_1, cat_2})
            
            
            
with open(f'/data/atlas/results/image_classification/pairwise_comparison/{activations_identifier}','wb') as f:
    pickle.dump(performance_dict,f)
    
print(f'pairwaise performance is saved as {activations_identifier}')