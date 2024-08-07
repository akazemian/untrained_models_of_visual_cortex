import gc
import logging

import numpy as np

from code_.model_activations.models.utils import load_model, load_full_identifier
from code_.model_activations.activation_extractor import Activations
from code_.encoding_score.regression.get_betas import NeuralRegression
from code_.encoding_score.regression.scores_tools import get_bootstrap_rvalues
from config import setup_logging

setup_logging()
MODEL_NAME = 'expansion_linear'
ANALYSIS = 'activation_function'
N_BOOTSTRAPS = 1000

def linear_models_(dataset, cfg, batch_size, device):

    N_ROWS = cfg[dataset]['test_data_size']
    ALL_SAMPLED_INDICES = np.random.choice(N_ROWS, (N_BOOTSTRAPS, N_ROWS), replace=True) # Sample indices for all 
    
    for features in cfg[dataset]['analysis'][ANALYSIS]['features']:
        
        # get model identifier
        activations_identifier = load_full_identifier(model_name=MODEL_NAME, features=features, 
                                                      layers=cfg[dataset]['analysis'][ANALYSIS]['layers'], 
                                                      dataset=dataset)
        logging.info(f"Model: {MODEL_NAME}, Features: {features}, Region: {cfg[dataset]['regions']}")
        # load model
        model = load_model(model_name=MODEL_NAME, 
                           features=features, 
                           layers=cfg[dataset]['analysis'][ANALYSIS]['layers'],
                           device=device)

        # extract activations 
        Activations(model=model, 
                    dataset=dataset, 
                    device=device,
                    batch_size=batch_size).get_array(activations_identifier) 

        logging.info(f"Predicting neural data from model activations")
        # predict neural data in a cross validated manner
        NeuralRegression(activations_identifier=activations_identifier,
                   dataset=dataset,
                   region=cfg[dataset]['regions'],
                   device= device).predict_data()

    logging.info(f"Getting a bootstrap distribution of scores")
    # get a bootstrap distribution of r-values between predicted and actual neural responses
    get_bootstrap_rvalues(model_name = MODEL_NAME,
            features=cfg[dataset]['analysis'][ANALYSIS]['features'],
            layers=cfg[dataset]['analysis'][ANALYSIS]['layers'],
            dataset=dataset, 
            subjects = cfg[dataset]['subjects'],
            region=cfg[dataset]['regions'],
            all_sampled_indices=ALL_SAMPLED_INDICES,
            device=device,
            file_name = 'linear_model')
    gc.collect()
