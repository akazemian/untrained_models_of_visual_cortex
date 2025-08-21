import numpy as np
import gc
import argparse
import os

from code_.encoding_score.regression.get_betas import EncodingScore
from code_.model_activations.activation_extractor import Activations
from code_.model_activations.loading import load_model, load_full_identifier
from code_.encoding_score.regression.scores_tools import get_bootstrap_data
from code_.eigen_analysis.compute_pcs import compute_model_pcs
from code_.model_activations.configs import model_cfg as cfg     
from config import CACHE

"""
This script is for obtaining the number of PCs required for explaining 85% of variance in the data for each model. Since this 
number differes for each model and each dataset, and given compute and storage limitations, for each model we save a fixed number of PCs, which 
is lower than max # PCs, and use this when plotting figure S5. For the majajhing dataset, this number is 2000, and for the naturalscenes dataset, 
its 7000. Further, we use inceremntal PCA from sklearn when computing 7000 PCs from the naturalscenes dataset.
"""


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
        default="cpu",
        help="Device to use (e.g., 'cuda' or 'cpu')"
        )
        args = parser.parse_args()

        MODELS =  ['fully_connected', 'vit' , 'expansion']

        if args.dataset == 'majajhong':
                TOTAL_COMPONENTS = 2000  
                INCREMENTAL = False

        elif args.dataset == 'naturalscenes':
                TOTAL_COMPONENTS = 70000  
                INCREMENTAL = True                
                
        for model_name in MODELS:
                
                for features in cfg[args.dataset]['models'][model_name]['features']:

                        pca_identifier = load_full_identifier(model_name=model_name, 
                                                        features=features, 
                                                        layers=cfg[args.dataset]['models'][model_name]['layers'], 
                                                        dataset=args.dataset,
                                                        principal_components = TOTAL_COMPONENTS) 
                        print('pca iden', pca_identifier)
                        if not os.path.exists(os.path.join(CACHE, 'pca', pca_identifier)):
                                print(f'computing PCs for {pca_identifier}')
                                compute_model_pcs(model_name, 
                                                  features, 
                                                  cfg[args.dataset]['models'][model_name]['layers'], 
                                                  args.batch_size,
                                                  args.dataset, 
                                                  TOTAL_COMPONENTS, 
                                                  args.device, 
                                                  incremental=INCREMENTAL)
                
        gc.collect()

                

if __name__ == "__main__":
        main()






