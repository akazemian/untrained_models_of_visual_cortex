import numpy as np
import gc
import argparse
import os

from code_.encoding_score.regression.get_betas import EncodingScore
from code_.model_activations.activation_extractor import Activations
from code_.model_activations.loading import load_model, load_full_identifier
from code_.encoding_score.regression.scores_tools import get_bootstrap_data
from code_.eigen_analysis.compute_pcs import compute_model_pcs
from code_.model_activations.configs import analysis_cfg as cfg     
from dotenv import load_dotenv

load_dotenv()

CACHE = os.getenv("CACHE")

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
        help="Name of the model (eg., expansion)"
        )
        parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (e.g., 'cuda' or 'cpu')"
        )
        args = parser.parse_args()

        N_BOOTSTRAPS = 1000
        N_ROWS = cfg[args.dataset]['test_data_size']
        ALL_SAMPLED_INDICES = np.random.choice(N_ROWS, (N_BOOTSTRAPS, N_ROWS), replace=True) # Sample indices for all 
        TOTAL_COMPONENTS = 1000  
        INCREMENTAL = False
        # TOTAL_COMPONENTS = 2000  
        BATCH_SIZE = 16
                
        for features in cfg[args.dataset]['analysis']['pca']['features']:

                pca_identifier = load_full_identifier(model_name='expansion', 
                                                        features=features, 
                                                        layers=cfg[args.dataset]['analysis']['pca']['layers'], 
                                                        dataset=args.dataset,
                                                        principal_components = TOTAL_COMPONENTS) 
                print('pca iden', pca_identifier)
                if not os.path.exists(os.path.join(CACHE, 'pca', pca_identifier)):
                        print('computing total PCs first')
                        compute_model_pcs(args.model, features, 
                                                cfg[args.dataset]['analysis']['pca']['layers'], BATCH_SIZE,
                                                args.dataset, TOTAL_COMPONENTS, 'cpu', incremental=INCREMENTAL)
                
                # set the number of components, if features are on the order of 10^2, compoennts are also max = 100, otherwise components = 1000
                N_COMPONENTS = [1,10,100] if features*36 <= 100 else [1,10,100,1000] 
                
                for n_components in N_COMPONENTS:
                
                        activations_identifier = load_full_identifier(model_name=args.model, 
                                                        features=features, 
                                                        layers=cfg[args.dataset]['analysis']['pca']['layers'], 
                                                        dataset=args.dataset,
                                                        principal_components = n_components)            
                        
                        print(features, n_components, activations_identifier)
                                
                        model = load_model(model_name=args.model, 
                                        features=features, 
                                        layers=cfg[args.dataset]['analysis']['pca']['layers'],
                                        device=args.device)
                        
                        Activations(model=model,
                                layer_names=['last'],
                                dataset=args.dataset,
                                device= args.device,
                                hook='pca',
                                pca_iden = pca_identifier,
                                n_components=n_components,
                                batch_size = BATCH_SIZE).get_array(activations_identifier) 

                        EncodingScore(activations_identifier=activations_identifier,
                                dataset=args.dataset,
                                region=cfg[args.dataset]['regions'],
                                device= args.device).get_scores()

                        gc.collect()

        get_bootstrap_data(model_name= args.model,
                        features=cfg[args.dataset]['analysis']['pca']['features'],
                        layers = cfg[args.dataset]['analysis']['pca']['layers'],
                        principal_components=N_COMPONENTS,
                        dataset=args.dataset, 
                        subjects=cfg[args.dataset]['subjects'],
                        file_name = 'pca',
                        region=cfg[args.dataset]['regions'],
                        all_sampled_indices=ALL_SAMPLED_INDICES,
                        device=args.device)
        gc.collect()

                

if __name__ == "__main__":
        main()






