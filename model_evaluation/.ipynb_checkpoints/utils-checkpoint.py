import sys
import os
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
from config import CACHE    

def get_activations_iden(model_info, dataset):
    
        model_name = model_info['iden'] 
        
        activations_identifier = model_name + '_' + f'{model_info["num_layers"]}_layers' + '_' + f'{model_info["num_features"]}_features' 

        if model_info['gpool'] == False:
            activations_identifier = activations_identifier + '_gpool=False'        
        
        if model_info['hook'] == 'pca':
            activations_identifier = activations_identifier + '_principal_components'
        
        return activations_identifier + '_' + dataset


def get_scores_iden(keys, gpool):
    models = []
    exclude_keys = []
    if gpool == False:
        exclude_keys.append('gpool=False')
    scores_path = os.path.join(CACHE,'encoding_scores_torch')
    for iden in os.listdir(scores_path):
        if all(sub in iden for sub in keys) and not any(exclude in iden for exclude in exclude_keys):
            return iden


def get_best_layer_iden(model_name,dataset,region,gpool):
    
    iden = model_name
    
    if gpool == False:
        iden = iden + '_' + 'gpool=False'
        
    iden = iden + '_' + dataset + '_' + region 
    
    return iden
    



