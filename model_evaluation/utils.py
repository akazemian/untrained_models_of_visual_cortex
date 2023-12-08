import sys
import os
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
from config import CACHE    

def get_activations_iden(model_info, dataset, *args, **kwargs):
    
        model_name = model_info['iden'] 
        
        if model_name == 'scat_transform':
            return get_st_activations_iden(model_info, dataset, *args, **kwargs)
        
        
        activations_identifier = model_name + '_' + f'{model_info["num_layers"]}_layers' + '_' + f'{model_info["num_features"]}_features' 

        if model_info['gpool'] == False:
            activations_identifier = activations_identifier + '_gpool=False'        
        
        if model_info['hook'] == 'pca':
            return activations_identifier + '_' + dataset + '_principal_components'                
                
        else:
            return activations_identifier + '_' + dataset
    
    
    
    
def get_st_activations_iden(model_info, dataset, J, L=8, M=224, N=224, random_proj = None):
    
        if model_info['gpool']:
            activations_identifier = 'scat_transform' + '_'+ f'J={J}_L={L}_M={M}_N={N}' 

        else:
            if random_proj is not None:
                activations_identifier = 'scat_transform' + '_' + f'randproj={RANDOM_PROJ}' + '_' + f'J={J}_L={L}_M={M}_N={N}' + '_' + 'gpool=False'      
            else:
                activations_identifier = 'scat_transform' + '_' +  f'J={J}_L={L}_M={M}_N={N}' + '_' + 'gpool=False' 

        
        if model_info['hook'] == 'pca':
            return activations_identifier + '_' + dataset + '_principal_components'               
            
        else:
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
    



