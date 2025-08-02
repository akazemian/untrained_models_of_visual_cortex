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

        MODEL_NAME = 'expansion'
        ANALYSIS = 'init_types'
        N_BOOTSTRAPS = 1000
        N_ROWS = cfg[args.dataset]['test_data_size']
        ALL_SAMPLED_INDICES = np.random.choice(N_ROWS, (N_BOOTSTRAPS, N_ROWS), replace=True) # Sample indices for all 

        for init_type in cfg[args.dataset]['analysis'][ANALYSIS]['variations']:
        
                for features in cfg[args.dataset]['analysis'][ANALYSIS]['features']:
                                
                        activations_identifier = load_full_identifier(model_name=MODEL_NAME,
                                                                features=features, 
                                                                layers=cfg[args.dataset]['analysis'][ANALYSIS]['layers'], 
                                                                dataset=args.dataset, 
                                                                init_type=init_type)
                        print(args.dataset, cfg[args.dataset]['regions'], activations_identifier)
                        
                        model = Expansion5L(filters_5 = features, 
                                                init_type=init_type).build()


                        Activations(model=model,
                                layer_names=['last'],
                                dataset=args.dataset,
                                device= args.device,
                                batch_size = 5).get_array(activations_identifier) 


                        EncodingScore(activations_identifier=activations_identifier,
                                dataset=args.dataset,
                                region=cfg[args.dataset]['regions'],
                                device= 'cpu').get_scores(iden= activations_identifier + '_' + cfg[args.dataset]['regions'])

                        gc.collect()



        get_bootstrap_data(model_name= MODEL_NAME,
                        features=cfg[args.dataset]['analysis'][ANALYSIS]['features'],
                        layers = cfg[args.dataset]['analysis'][ANALYSIS]['layers'],
                        dataset=args.dataset, 
                        subjects=cfg[args.dataset]['subjects'],
                        init_types=cfg[args.dataset]['analysis'][ANALYSIS]['variations'],
                        file_name = ANALYSIS,
                        region=cfg[args.dataset]['regions'],
                        all_sampled_indices=ALL_SAMPLED_INDICES,
                        device='cpu'
                        )    
if __name__ == "__main__":
    main()
