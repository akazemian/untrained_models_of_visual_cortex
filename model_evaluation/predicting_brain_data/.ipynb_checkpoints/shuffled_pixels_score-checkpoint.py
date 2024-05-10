from model_evaluation.predicting_brain_data.regression.scorer import EncodingScore
from model_features.activation_extractor import Activations
from model_features.models.models import load_full_iden
from model_features.models.models import load_model, load_full_iden
from configs import analysis_cfg as cfg
from model_evaluation.results.predicting_brain_data.tools import get_bootstrap_data

import gc
import numpy as np

DATASET = 'naturalscenes_shuffled' #'majajhong'
MODEL_NAME = 'expansion'
N_BOOTSTRAPS = 1000
N_ROWS = cfg[DATASET]['test_data_size']
ALL_SAMPLED_INDICES = np.random.choice(N_ROWS, (N_BOOTSTRAPS, N_ROWS), replace=True) # Sample indices for all 


for features in cfg[DATASET]['models'][MODEL_NAME]['features']:
        
    activations_identifier = load_full_iden(model_name=MODEL_NAME,
                                            features=features, 
                                            layers=cfg[DATASET]['models'][MODEL_NAME]['layers'], 
                                            dataset=DATASET)
    
    print(DATASET, cfg[DATASET]['regions'], activations_identifier)
    
    model = load_model(model_name=MODEL_NAME, features=features, layers=cfg[DATASET]['models'][MODEL_NAME]['layers'])    
    
    Activations(model=model,
            layer_names=['last'],
            dataset=DATASET,
            device= 'cuda',
            batch_size = 50).get_array(activations_identifier) 
    
    
    EncodingScore(activations_identifier=activations_identifier,
               dataset=DATASET,
               region=cfg[DATASET]['regions'],
               device= 'cpu').get_scores(iden= activations_identifier + '_' + cfg[DATASET]['regions'])


get_bootstrap_data(model_name= MODEL_NAME,
        features=cfg[DATASET]['models'][MODEL_NAME]['features'],
        layers = cfg[DATASET]['models'][MODEL_NAME]['layers'],
        dataset=DATASET,
        subjects=cfg[DATASET]['subjects'],
        file_name = 'suffleds-pixels',
        region=cfg[DATASET]['regions'],
        all_sampled_indices=ALL_SAMPLED_INDICES,
        device='cpu'
          )    

