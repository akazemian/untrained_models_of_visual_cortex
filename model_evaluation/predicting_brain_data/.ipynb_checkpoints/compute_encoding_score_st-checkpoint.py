import os 
import sys
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
import warnings
warnings.filterwarnings('ignore')

from image_tools.processing import *
from model_evaluation.predicting_brain_data.regression.scorer import EncodingScore
from model_evaluation.utils import get_activations_iden
from model_features.activation_extractor import Activations
from model_features.models.models import load_model_dict
import gc
from model_features.models.scat_transform import ScatTransformKymatio as st

# define local variables
# DATASET = 'majajhong'
# REGIONS = ['V4','IT']


DATASET = 'naturalscenes'
REGIONS = ['general']

DEVICE = 'cuda' 
MAX_POOL = None
RANDOM_PROJ = None
GLOBAL_POOL = True 
M, N = 224, 224
# try max_order=3

for J in [5]:   
    
    for L in [8]:
        
        for region in REGIONS:

            if GLOBAL_POOL:
                activations_identifier = 'scat_transfom' + '_'+ f'J={J}_L={L}_M={M}_N={N}' + '_' + DATASET
                
            elif RANDOM_PROJ is not None:
                activations_identifier = 'scat_transfom' + '_' + f'randproj={RANDOM_PROJ}' + '_' + f'J={J}_L={L}_M={M}_N={N}' + '_' + 'gpool=False' + '_' + DATASET                
            
            else:
                activations_identifier = 'scat_transfom' + '_' + f'maxpool' + '_' +  f'J={J}_L={L}_M={M}_N={N}' + '_' + 'gpool=False' + '_' + DATASET

            print(activations_identifier)
            
            Activations(model=st(J=J, L=L, M=M, N=N, random_proj = RANDOM_PROJ, max_pool = MAX_POOL, global_pool = GLOBAL_POOL, device = DEVICE).Build(),
                        layer_names=['last'],
                        dataset=DATASET,
                        device= DEVICE,
                        batch_size = 100,
                        image_size = M,
                        compute_mode = 'fast').get_array(activations_identifier)   


            scores_iden = activations_identifier + '_' + region

            EncodingScore(activations_identifier=activations_identifier,
                           dataset=DATASET,
                           region=region,
                           device= 'cpu').get_scores(scores_iden)
            gc.collect()


