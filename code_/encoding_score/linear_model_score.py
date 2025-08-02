import numpy as np
import gc
import argparse

from code_.encoding_score.regression.get_betas import EncodingScore
from code_.model_activations.activation_extractor import Activations
from code_.model_activations.loading import load_model, load_full_identifier
from code_.encoding_score.regression.scores_tools import get_bootstrap_data
from code_.model_activations.configs import model_cfg as cfg      


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
    
    model_name = 'expansion_linear'
    N_BOOTSTRAPS = 1000
    N_ROWS = cfg[args.dataset]['test_data_size']
    ALL_SAMPLED_INDICES = np.random.choice(N_ROWS, (N_BOOTSTRAPS, N_ROWS), replace=True) # Sample indices for all 
    BATCH_SIZE = 100

    for region in cfg[args.dataset]['regions']:
            
        if region not in ['ventral visual stream', 'IT']:
            pass
        else:
            for features in cfg[args.dataset]['models'][model_name]['features']:
    
                activations_identifier = load_full_identifier(model_name=model_name, features=features, 
                                                                layers=cfg[args.dataset]['models'][model_name]['layers'], 
                                                                dataset=args.dataset)
                print(args.dataset, region, activations_identifier)
                
                model = load_model(model_name=model_name, 
                                   features=features, 
                                   layers=cfg[args.dataset]['models'][model_name]['layers'])

                Activations(model=model,
                        layer_names=['last'],
                        dataset=args.dataset,
                        device= args.device,
                        batch_size = BATCH_SIZE).get_array(activations_identifier) 


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
                    file_name = model_name,
                    region=region,
                    all_sampled_indices=ALL_SAMPLED_INDICES,
                    device=args.device)


if __name__ == "__main__":
    main()

            
            





        
        

