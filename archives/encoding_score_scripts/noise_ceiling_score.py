import argparse
import gc
import numpy as np

from code_.encoding_score.regression.get_betas import EncodingScore
from code_.model_activations.configs import model_cfg as cfg      
from code_.encoding_score.regression.scores_tools import get_nc_score


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
        default="cpu",
        help="Device to use (e.g., 'cuda' or 'cpu')"
    )
    args = parser.parse_args()
    activations_identifier = "noise_ceiling"

    N_BOOTSTRAPS = 1000
    N_ROWS = cfg[args.dataset]['test_data_size']
    if args.dataset == 'naturalscenes':
        N_ROWS = 175
    ALL_SAMPLED_INDICES = np.random.choice(N_ROWS, (N_BOOTSTRAPS, N_ROWS), replace=True) # Sample indices for all 


    for region in cfg[args.dataset]["regions"]:
        EncodingScore(
            activations_identifier=activations_identifier,
            dataset=args.dataset,
            region=region,
            device=args.device,
            nc=True
        ).get_scores()
        
        get_nc_score(activations_identifier, 
                     cfg[args.dataset]['subjects'], 
                     args.dataset, 
                     region,
                     ALL_SAMPLED_INDICES)
        
        
        gc.collect()


if __name__ == "__main__":
    main()
