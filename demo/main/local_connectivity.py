import numpy as np
import gc
import argparse

from code_.encoding_score.regression.get_betas import EncodingScore
from code_.model_activations.activation_extractor import Activations
from code_.model_activations.loading import load_model, load_full_identifier
from code_.encoding_score.regression.scores_tools import get_bootstrap_data
from code_.model_activations.configs import analysis_cfg as cfg      
from code_.tools.utils import timeit, setup_logging

import gc
import numpy as np


# define local variables
@timeit
def main():
        parser = argparse.ArgumentParser(
        description="Compute encoding scores for all regions in a dataset"
        )
        parser.add_argument(
        "--model",
        type=str,
        default='expansion',
        help="Name of the model"
        )
        parser.add_argument(
        "--dataset",
        type=str,
        default='majajhong_demo_shuffled',
        help="Name of the dataset (e.g., 'majajhong')"
        )
        parser.add_argument(
        "--batch_size",
        type=str,
        default=50,
        help="Name of the dataset (e.g., 'majajhong')"
        )
        parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (e.g., 'cuda' or 'cpu')"
        )

        args = parser.parse_args()

        N_BOOTSTRAPS = 100
        N_ROWS = cfg[args.dataset]['test_data_size']
        ALL_SAMPLED_INDICES = np.random.choice(N_ROWS, (N_BOOTSTRAPS, N_ROWS), replace=True) # Sample indices for all 


        for features in cfg[args.dataset]['models'][args.model]['features']:
                
                activations_identifier = load_full_identifier(model_name=args.model,
                                                        features=features, 
                                                        layers=cfg[args.dataset]['models'][args.model]['layers'], 
                                                        dataset=args.dataset)
                
                print(args.dataset, cfg[args.dataset]['regions'], activations_identifier)
                
                model = load_model(model_name=args.model, features=features, layers=cfg[args.dataset]['models'][args.model]['layers'])    
                
                Activations(model=model,
                        layer_names=['last'],
                        dataset=args.dataset,
                        device= args.device,
                        batch_size = int(args.batch_size)).get_array(activations_identifier) 
                
                
                EncodingScore(activations_identifier=activations_identifier,
                        dataset=args.dataset,
                        region=cfg[args.dataset]['regions'],
                        device= args.device).get_scores()


        get_bootstrap_data(model_name= args.model,
                features=cfg[args.dataset]['models'][args.model]['features'],
                layers = cfg[args.dataset]['models'][args.model]['layers'],
                dataset=args.dataset,
                subjects=cfg[args.dataset]['subjects'],
                file_name = args.model,
                region=cfg[args.dataset]['regions'],
                all_sampled_indices=ALL_SAMPLED_INDICES,
                device=args.device,
                n_bootstraps=N_BOOTSTRAPS,
                )    

if __name__ == "__main__":
    main()

            