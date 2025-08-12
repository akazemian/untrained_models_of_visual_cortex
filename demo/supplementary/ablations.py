'''
no_sp: to disrupt spatial contiguity at the input to expansion layer, and determine if the locality of 
the operation in expansion layer is critical; 

lrp: linear + ReLU projection on the incoming feature tensor, instead of conv + ReLU + pooling, 
to see if a linear random projection from the incoming layer is already enough, or if one needs to process it 
further with convolution; 

no_relu: just random convolution + average pooling at the expansion layer, to see if that suffices. 
It will be intriguing if the high-dimensional random 3x3 convolution is the key for the improved performance 
in CNN models. One guess is that direct (linear projection) dimension expansion from feature tensor will 
not help too much.
'''
import argparse
import gc
import numpy as np      

from code_.model_activations.models.expansion_lrp import ExpansionLRP
from code_.model_activations.models.expansion_no_relu import ExpansionNoReLU
from code_.model_activations.models.expansion_no_sp import ExpansionNoSP
from code_.model_activations.models.expansion_256 import Expansion256

from code_.tools.processing import *
from code_.encoding_score.regression.get_betas import EncodingScore
from code_.model_activations.activation_extractor import Activations

from code_.encoding_score.benchmarks.nsd import load_nsd_data
from code_.model_activations.loading import load_model, load_full_identifier
from code_.model_activations.configs import model_cfg as cfg
from code_.encoding_score.regression.scores_tools import get_bootstrap_data
from code_.tools.utils import timeit, setup_logging


@timeit
def main():
    parser = argparse.ArgumentParser(
        description="Compute encoding scores for all regions in a dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default='expansion',
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='majajhong_demo',
        help="Name of the dataset (e.g., 'majajhong')"
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
    
    
    N_BOOTSTRAPS = 100
    N_ROWS = cfg[args.dataset]['test_data_size']
    ALL_SAMPLED_INDICES = np.random.choice(N_ROWS, (N_BOOTSTRAPS, N_ROWS), replace=True) # Sample indices for all 
    variations = ['256_features','lrp', 'no_relu', 'no_sp']

    for variation in variations:
        for region in cfg[args.dataset]['regions']:
            
            for features in cfg[args.dataset]['models'][args.model]['features']:

                activations_identifier = load_full_identifier(model_name=args.model, 
                                                            features=features, 
                                                            layers=cfg[args.dataset]['models'][args.model]['layers'], 
                                                            dataset=args.dataset)
                
                activations_identifier  = activations_identifier + '_' + variation 
                print(args.dataset, region, activations_identifier)
                
                match variation:
                    case 'lrp':
                        print('reading this model')
                        model = ExpansionLRP(filters_5=features).build()
                    case 'no_relu':
                        print('reading no relu model')
                        model = ExpansionNoReLU(filters_5=features).build()
                    case 'no_sp':
                        print('reading no sp model')
                        model = ExpansionNoSP(filters_5=features).build()
                    case '256_features':
                        model = Expansion256(filters_5=features).build()

                Activations(model=model,
                        layer_names=['last'],
                        dataset=args.dataset,
                        device= args.device,
                        batch_size = int(args.batch_size)).get_array(activations_identifier) 

                EncodingScore(activations_identifier=activations_identifier,
                        dataset=args.dataset,
                        region=region,
                        device= args.device).get_scores(iden= activations_identifier + '_' + region)
                gc.collect()

        
            get_bootstrap_data(model_name= args.model,
                    features=cfg[args.dataset]['models'][args.model]['features'],
                    layers = cfg[args.dataset]['models'][args.model]['layers'],
                    dataset=args.dataset, 
                    subjects = cfg[args.dataset]['subjects'],
                    extension = f'_{variation}',
                    file_name = args.model + '_' + variation,
                    region=region,
                    all_sampled_indices=ALL_SAMPLED_INDICES,
                    device=args.device,
                    n_bootstraps=N_BOOTSTRAPS)
                

if __name__ == "__main__":
    main()


            
            

