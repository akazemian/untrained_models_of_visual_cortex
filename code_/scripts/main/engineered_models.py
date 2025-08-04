import numpy as np
import gc
import argparse

from code_.encoding_score.regression.get_betas import EncodingScore
from code_.model_activations.activation_extractor import Activations
from code_.model_activations.loading import load_model, load_full_identifier
from code_.encoding_score.regression.scores_tools import get_bootstrap_data
from code_.model_activations.configs import model_cfg as cfg      
from code_.tools.utils import timeit, setup_logging


# define local variables
@timeit
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
        "--batch_size",
        default=5,
        type=str,
        help="Batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (e.g., 'cuda' or 'cpu')"
    )
    args = parser.parse_args()
    
    MODELS = ['fully_connected', 'vit' , 'expansion', 'fully_random', 'expansion_linear']
    N_BOOTSTRAPS = 1000
    N_ROWS = cfg[args.dataset]['test_data_size']
    ALL_SAMPLED_INDICES = np.random.choice(N_ROWS, (N_BOOTSTRAPS, N_ROWS), replace=True) # Sample indices for all 


    for region in cfg[args.dataset]['regions']:
            
        for model_name in MODELS:
            
            if (model_name in ['expansion_linear','fully_random']) and (region not in ['ventral visual stream', 'IT']):
                continue

            if ('suffled' in args.dataset) and (region not in ['ventral visual stream', 'IT']) and (model != 'expansion'):
                continue
            
            else:
                for features in cfg[args.dataset]['models'][model_name]['features']:
        
                    activations_identifier = load_full_identifier(model_name=model_name, features=features, 
                                                                  layers=cfg[args.dataset]['models'][model_name]['layers'], 
                                                                  dataset=args.dataset)
                    print(args.dataset, region, activations_identifier)
                    
                    model = load_model(model_name=model_name, features=features, layers=cfg[args.dataset]['models'][model_name]['layers'])

                    Activations(model=model,
                            layer_names=['last'],
                            dataset=args.dataset,
                            device= args.device,
                            batch_size = int(args.batch_size)).get_array(activations_identifier) 


                    EncodingScore(activations_identifier=activations_identifier,
                            dataset=args.dataset,
                            region=region,
                            device= args.device).get_scores()
                    gc.collect()

            
            get_bootstrap_data(model_name= model_name,
                    features=cfg[args.dataset]['models'][model_name]['features'],
                    layers = cfg[args.dataset]['models'][model_name]['layers'],
                    dataset=args.dataset, 
                    subjects = cfg[args.dataset]['subjects'],
                    file_name = 'TEST_BITCH_' + model_name,
                    region=region,
                    all_sampled_indices=ALL_SAMPLED_INDICES,
                    device=args.device,
                    n_bootstraps=N_BOOTSTRAPS,
                    )


if __name__ == "__main__":
    main()

            
            





        
        

