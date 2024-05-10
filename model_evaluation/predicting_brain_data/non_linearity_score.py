from model_evaluation.predicting_brain_data.regression.scorer import EncodingScore
from model_features.activation_extractor import Activations
from model_features.models.models import load_full_iden
from configs import analysis_cfg as cfg
import numpy as np
import gc
from model_features.models.expansion import Expansion5L
from model_evaluation.results.predicting_brain_data.tools import get_bootstrap_data

DATASET = 'naturalscenes' #'majajhong'
MODEL_NAME = 'expansion'
ANALYSIS = 'non_linearities'
N_BOOTSTRAPS = 1000
N_ROWS = cfg[DATASET]['test_data_size']
ALL_SAMPLED_INDICES = np.random.choice(N_ROWS, (N_BOOTSTRAPS, N_ROWS), replace=True) # Sample indices for all 



for features in cfg[DATASET]['analysis'][ANALYSIS]['features']:
    for non_linearity in cfg[DATASET]['analysis'][ANALYSIS]['variations']:
                    
            activations_identifier = load_full_iden(model_name=MODEL_NAME,
                                                    features=features, 
                                                    layers=cfg[DATASET]['analysis'][ANALYSIS]['layers'], 
                                                    dataset=DATASET, 
                                                    non_linearity=non_linearity)
            print(DATASET, cfg[DATASET]['regions'], activations_identifier)
            
            model = Expansion5L(filters_5 = features, 
                                non_linearity=non_linearity).Build()


            Activations(model=model,
                    layer_names=['last'],
                    dataset=DATASET,
                    device= 'cuda',
                    batch_size = 50).get_array(activations_identifier) 


            EncodingScore(activations_identifier=activations_identifier,
                       dataset=DATASET,
                       region=cfg[DATASET]['regions'],
                       device= 'cuda').get_scores(iden= activations_identifier + '_' + cfg[DATASET]['regions'])

            gc.collect()



get_bootstrap_data(model_name= MODEL_NAME,
                    features=cfg[DATASET]['analysis'][ANALYSIS]['features'],
                    layers = cfg[DATASET]['analysis'][ANALYSIS]['layers'],
                    dataset=DATASET, 
                    subjects=cfg[DATASET]['subjects'],
                    non_linearities=cfg[DATASET]['analysis'][ANALYSIS]['variations'],
                    file_name = ANALYSIS,
                    region=cfg[DATASET]['regions'],
                    all_sampled_indices=ALL_SAMPLED_INDICES,
                    device='cpu'
                      )    

