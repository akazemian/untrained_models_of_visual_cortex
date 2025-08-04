import gc
import logging

import numpy as np 
import argparse
from code_.encoding_score.regression.get_betas import EncodingScore
from code_.model_activations.activation_extractor import Activations
from code_.model_activations.loading import load_model, load_full_identifier
from code_.encoding_score.regression.scores_tools import get_bootstrap_data
from code_.model_activations.configs import model_cfg as cfg

from config import setup_logging

setup_logging()

layers_dict = {'alexnet_trained': [i for i in range(1,6)],
               'alexnet_untrained': [i for i in range(1,6)],
               'vit_untrained': [i for i in range(12)],
               'vit_trained':[i for i in range(12)],
               'resnet50_trained': [i for i in range(1,5)]}

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
        "--model",
        type=str,
        required=True,
        help="Name of the model (e.g., 'alexnet_trained')"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (e.g., 'cuda' or 'cpu')"
    )
    args = parser.parse_args()
    
    N_BOOTSTRAPS = 1000
    N_ROWS = cfg[args.dataset]['test_data_size']
    ALL_SAMPLED_INDICES = np.random.choice(N_ROWS, (N_BOOTSTRAPS, N_ROWS), replace=True) # Sample indices for all 

    for region in cfg[args.dataset]['regions']:
                        
        indintifier_list = []
    
        for layer_num in layers_dict[args.model]:
                    
            activations_identifier = load_full_identifier(model_name=args.model, 
                                                        layers=layer_num, 
                                                        dataset=args.dataset)
            print(activations_identifier)
            indintifier_list.append(activations_identifier)
            
                    
            model = load_model(model_name=args.model, 
                            layers=layer_num,
                            device=args.device)
            
            Activations(model=model, 
                        dataset=args.dataset, 
                        device= args.device).get_array(activations_identifier) 
        
        logging.info(f"Predicting neural data from model activations")
        # predict neural data from the best layer's activations in a cross validated manner
        EncodingScore(activations_identifier=indintifier_list,
                    dataset=args.dataset,
                    region=region,
                    device= 'cpu',
                    best_layer=True).get_scores()

        logging.info(f"Getting a bootstrap distribution of scores")
        get_bootstrap_data(model_name= args.model,
                        file_name=args.model,
                            features=[None], 
                            layers = None,
                            dataset=args.dataset, 
                            subjects = cfg[args.dataset]['subjects'],
                            region=region, 
                            all_sampled_indices=ALL_SAMPLED_INDICES,
                            device=args.device)

        gc.collect()               


if __name__ == "__main__":
    main()

