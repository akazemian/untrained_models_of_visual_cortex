import gc
import logging

import numpy as np 

from code_.encoding_score.regression.get_betas import EncodingScore
from code_.model_activations.activation_extractor import Activations
from code_.model_activations.loading import load_model, load_full_identifier
from code_.encoding_score.regression.scores_tools import get_bootstrap_data
from code_.model_activations.configs import model_cfg as cfg

from config import setup_logging

setup_logging()

# define local variables
# DATASET = 'majajhong'
DATASET = 'naturalscenes'

MODEL_NAME = 'resnet50'    
N_BOOTSTRAPS = 1000

device = 'cuda'
batch_size = 64

N_ROWS = cfg[DATASET]['test_data_size']
ALL_SAMPLED_INDICES = np.random.choice(N_ROWS, (N_BOOTSTRAPS, N_ROWS), replace=True) 

for region in cfg[DATASET]['regions']:
                    
    indintifier_list = []
    
    for layer_num in range(1,5):
                
        activations_identifier = load_full_identifier(model_name=MODEL_NAME, 
                                                      layers=layer_num, 
                                                      dataset=DATASET)
        print(activations_identifier)
        indintifier_list.append(activations_identifier)
        
                
        model = load_model(model_name=MODEL_NAME, 
                           layers=layer_num,
                           device=device)
        
        Activations(model=model, 
                    dataset=DATASET, 
                    device= device,
                    batch_size=batch_size).get_array(activations_identifier) 
    
    logging.info(f"Predicting neural data from model activations")
    # predict neural data from the best layer's activations in a cross validated manner
    EncodingScore(activations_identifier=indintifier_list,
                dataset=DATASET,
                region=region,
                device= 'cpu',
                best_layer=True).get_scores()

    logging.info(f"Getting a bootstrap distribution of scores")
    get_bootstrap_data(model_name= MODEL_NAME,
                       file_name=MODEL_NAME,
                        features=[None], 
                        layers = None,
                        dataset=DATASET, 
                        subjects = cfg[DATASET]['subjects'],
                        region=region, 
                        all_sampled_indices=ALL_SAMPLED_INDICES,
                        device=device)

    gc.collect()               





