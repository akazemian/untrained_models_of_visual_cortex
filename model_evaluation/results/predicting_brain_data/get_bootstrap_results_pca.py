from tools import *
import matplotlib.colors as mcolors
import gc
from model_features.models.models_config import cfg
PREDS_PATH = '/data/atlas/.cache/beta_predictions_new'
PATH_TO_BOOTSTRAP = '/home/akazemi3/Desktop/untrained_models_of_visual_cortex/model_evaluation/results/predicting_brain_data/bootstrap_data'
N_BOOTSTRAPS = 1000
DATASET = 'naturalscenes' #majahong
N_ROWS = cfg[DATASET]['test_data_size']
ALL_SAMPLED_INDICES = np.random.choice(N_ROWS, (N_BOOTSTRAPS, N_ROWS), replace=True) # Sample indices for all bootstraps at once




for region in cfg[DATASET]['regions']:
     
    print(region)
    # engineered models 
    # get_bootstrap_data(models= ['expansion','expansion_linear','fully_connected'],
    #                     features=cfg[DATASET]['models']['expansion']['features'],
    #                     layers = cfg[DATASET]['models']['expansion']['layers'],
    #                     dataset=dataset, 
    #                     subjects=cfg[DATASET]['subjects'],
    #                     file_name = f'engineered_new',
    #                     region=region,
    #                     all_sampled_indices=ALL_SAMPLED_INDICES,
    #                     batch_size=5,
    #                     n_bootstraps = N_BOOTSTRAPS,
    #                     device='cpu')
    # alexnet
    # get_bootstrap_data(models=['alexnet'],
    #                     features=[None],
    #                     dataset=dataset, 
    #                     subjects=info_dict['subjects'],
    #                     layers='best',
    #                     file_name='alexnet_new',
    #                     region=region,
    #                     all_sampled_indices=all_sampled_indices,
    #                   device='cpu')
    
    # ViT
    get_bootstrap_data(models= ['ViT'],
                        features=cfg[DATASET]['models']['ViT']['features'],
                        layers = cfg[DATASET]['models']['ViT']['layers'],
                        dataset=DATASET, 
                        subjects=cfg[DATASET]['subjects'],
                        file_name = f'ViT_new',
                        region=region,
                        all_sampled_indices=ALL_SAMPLED_INDICES,
                        batch_size=5,
                        n_bootstraps = N_BOOTSTRAPS,
                        device='cpu')
    gc.collect()


