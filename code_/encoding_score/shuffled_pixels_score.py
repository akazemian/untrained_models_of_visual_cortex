import numpy as np
import gc
import argparse

from code_.encoding_score.regression.get_betas import EncodingScore
from code_.model_activations.activation_extractor import Activations
from code_.model_activations.loading import load_model, load_full_identifier
from code_.encoding_score.regression.scores_tools import get_bootstrap_data
from code_.model_activations.configs import analysis_cfg as cfg      

import gc
import numpy as np


# define local variables

def main():
        parser = argparse.ArgumentParser(
        description="Compute encoding scores for all regions in a dataset"
        )
        parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset (e.g., 'majajhong')"
        )
        parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (e.g., 'cuda' or 'cpu')"
        )
        args = parser.parse_args()

        DATASET = args.dataset + '_shuffled' 
        MODEL_NAME = 'expansion'
        N_BOOTSTRAPS = 1000
        N_ROWS = cfg[args.dataset]['test_data_size']
        ALL_SAMPLED_INDICES = np.random.choice(N_ROWS, (N_BOOTSTRAPS, N_ROWS), replace=True) # Sample indices for all 


        for features in cfg[DATASET]['models'][MODEL_NAME]['features']:
                
                activations_identifier = load_full_identifier(model_name=MODEL_NAME,
                                                        features=features, 
                                                        layers=cfg[DATASET]['models'][MODEL_NAME]['layers'], 
                                                        dataset=DATASET)
                
                print(DATASET, cfg[DATASET]['regions'], activations_identifier)
                
                model = load_model(model_name=MODEL_NAME, features=features, layers=cfg[DATASET]['models'][MODEL_NAME]['layers'])    
                
                Activations(model=model,
                        layer_names=['last'],
                        dataset=DATASET,
                        device= args.device,
                        batch_size = 50).get_array(activations_identifier) 
                
                
                EncodingScore(activations_identifier=activations_identifier,
                        dataset=DATASET,
                        region=cfg[DATASET]['regions'],
                        device= args.device).get_scores(iden= activations_identifier + '_' + cfg[DATASET]['regions'])


        get_bootstrap_data(model_name= MODEL_NAME,
                features=cfg[DATASET]['models'][MODEL_NAME]['features'],
                layers = cfg[DATASET]['models'][MODEL_NAME]['layers'],
                dataset=DATASET,
                subjects=cfg[DATASET]['subjects'],
                file_name = model,
                region=cfg[DATASET]['regions'],
                all_sampled_indices=ALL_SAMPLED_INDICES,
                device=args.device
                )    

if __name__ == "__main__":
    main()

            