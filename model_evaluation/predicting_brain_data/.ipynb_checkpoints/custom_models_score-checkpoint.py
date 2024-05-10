from image_tools.processing import *
from model_evaluation.predicting_brain_data.regression.scorer import EncodingScore
from model_features.activation_extractor import Activations
import gc
from model_evaluation.predicting_brain_data.benchmarks.nsd import load_nsd_data
from model_features.models.models import load_model, load_full_iden
from configs import model_cfg as cfg
from model_evaluation.results.predicting_brain_data.tools import get_bootstrap_data
import numpy as np       


# define local variables

DATASET = 'naturalscenes' #'majajhong'
MODELS = ['fully_random'] #expansion, 'expansion_linear', 'fully_connected', 'vit'
N_BOOTSTRAPS = 1000
N_ROWS = cfg[DATASET]['test_data_size']
ALL_SAMPLED_INDICES = np.random.choice(N_ROWS, (N_BOOTSTRAPS, N_ROWS), replace=True) # Sample indices for all 




for region in cfg[DATASET]['regions']:
        
    for model_name in MODELS:
        
        if model_name in ['expansion_linear','fully_random'] and region != 'ventral visual stream':
            pass
        else:
            for features in cfg[DATASET]['models'][model_name]['features']:
    
                activations_identifier = load_full_iden(model_name=model_name, features=features, layers=cfg[DATASET]['models'][model_name]['layers'], dataset=DATASET)
                print(DATASET, region, activations_identifier)
                
                model = load_model(model_name=model_name, features=features, layers=cfg[DATASET]['models'][model_name]['layers'])

                Activations(model=model,
                        layer_names=['last'],
                        dataset=DATASET,
                        device= 'cuda',
                        batch_size = 50).get_array(activations_identifier) 


                EncodingScore(activations_identifier=activations_identifier,
                           dataset=DATASET,
                           region=region,
                           device= 'cpu').get_scores(iden= activations_identifier + '_' + region)
                gc.collect()

                
            get_bootstrap_data(model_name= model_name,
                    features=cfg[DATASET]['models'][model_name]['features'],
                    layers = cfg[DATASET]['models'][model_name]['layers'],
                    dataset=DATASET, 
                    subjects = cfg[DATASET]['subjects'],
                    file_name = model_name,
                    region=region,
                    all_sampled_indices=ALL_SAMPLED_INDICES,
                              device='cpu')
            gc.collect()
            
            





        
        

