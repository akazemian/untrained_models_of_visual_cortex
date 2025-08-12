import numpy as np
import gc
import argparse

from code_.encoding_score.regression.get_betas import EncodingScore
from code_.model_activations.activation_extractor import Activations
from code_.model_activations.loading import load_full_identifier
from code_.encoding_score.regression.scores_tools import get_bootstrap_data
from code_.model_activations.configs import analysis_cfg as cfg      
from code_.model_activations.models.expansion import Expansion5L
import gc
import numpy as np
from code_.tools.utils import timeit, setup_logging

@timeit
def main():
        parser = argparse.ArgumentParser(
        description="Compute encoding scores for all regions in a dataset"
        )
        parser.add_argument(
        "--dataset",
        type=str,
        default='majajhong_demo',
        help="Name of the dataset (e.g., 'majajhong')"
        )
        parser.add_argument(
        "--model",
        type=str,
        default='expansion',
        help="Name of the model (e.g., 'expansion')"
        )
        parser.add_argument(
        "--batch_size",
        default=50,
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

        MODEL_NAME = 'expansion'
        ANALYSIS = 'non_linearities'
        N_BOOTSTRAPS = 100
        N_ROWS = cfg[args.dataset]['test_data_size']
        ALL_SAMPLED_INDICES = np.random.choice(N_ROWS, (N_BOOTSTRAPS, N_ROWS), replace=True) # Sample indices for all 

        for features in cfg[args.dataset]['analysis'][ANALYSIS]['features']:
                for non_linearity in cfg[args.dataset]['analysis'][ANALYSIS]['variations']:
                                
                        activations_identifier = load_full_identifier(model_name=MODEL_NAME,
                                                                features=features, 
                                                                layers=cfg[args.dataset]['analysis'][ANALYSIS]['layers'], 
                                                                dataset=args.dataset, 
                                                                non_linearity=non_linearity)
                        print(args.dataset, cfg[args.dataset]['regions'], activations_identifier)
                        
                        model = Expansion5L(filters_5 = features, 
                                                non_linearity=non_linearity).build()


                        Activations(model=model,
                                layer_names=['last'],
                                dataset=args.dataset,
                                device=args.device,
                                batch_size = int(args.batch_size)).get_array(activations_identifier) 


                        EncodingScore(activations_identifier=activations_identifier,
                                dataset=args.dataset,
                                region=cfg[args.dataset]['regions'],
                                device='cpu').get_scores(iden= activations_identifier + '_' + cfg[args.dataset]['regions'])

                        gc.collect()



        get_bootstrap_data(model_name= MODEL_NAME,
                        features=cfg[args.dataset]['analysis'][ANALYSIS]['features'],
                        layers = cfg[args.dataset]['analysis'][ANALYSIS]['layers'],
                        dataset=args.dataset, 
                        subjects=cfg[args.dataset]['subjects'],
                        non_linearities=cfg[args.dataset]['analysis'][ANALYSIS]['variations'],
                        file_name = ANALYSIS,
                        region=cfg[args.dataset]['regions'],
                        all_sampled_indices=ALL_SAMPLED_INDICES,
                        device='cpu',
                        n_bootstraps=N_BOOTSTRAPS,

                        )    

if __name__ == "__main__":
    main()

            