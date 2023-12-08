import os 
import sys
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
import warnings
warnings.filterwarnings('ignore')

from image_tools.processing import *
from model_evaluation.predicting_brain_data.regression.scorer import EncodingScore
from model_evaluation.utils import get_st_activations_iden
from model_features.activation_extractor import Activations
from model_features.models.models import load_model_dict
import gc
from model_features.models.scat_transform import ScatTransformKymatio as st

# define local variables
# DATASET = 'majajhong'
# REGIONS = ['V4','IT']

MODEL_NAME = 'scat_transform_pcs'
DATASET = 'naturalscenes'
REGIONS = ['general']
DEVICE = 'cuda' 
MAX_POOL = True
RANDOM_PROJ = None
GLOBAL_POOL = False 
M, N = 224, 224


for J in [3]:   
    
    for L in [4]:
        
        for region in REGIONS:
                
            model_info = load_model_dict(name=MODEL_NAME, gpool=GLOBAL_POOL)
            
            if MODEL_NAME == 'scat_transform_pcs':
                os.system('python Desktop/random_models_of_visual_cortex/model_evaluation/eigen_analysis/compute_pcs.py')
            
#             activations_identifier = get_st_activations_iden(model_info=model_info, 
#                                                              dataset=DATASET, 
#                                                              random_proj = RANDOM_PROJ,
#                                                              J=J, L=L, M=M, N=N)               
#             print(activations_identifier)
            
#             Activations(model=st(J=J, L=L, M=M, N=N, 
#                                  random_proj = RANDOM_PROJ, 
#                                  max_pool = MAX_POOL, 
#                                  global_pool = GLOBAL_POOL, 
#                                  device = DEVICE).Build(),
#                         hook= model_info['hook'],
#                         layer_names=['last'],
#                         dataset=DATASET,
#                         device= DEVICE,
#                         batch_size = 200,
#                         image_size = M,
#                         compute_mode = 'fast').get_array(activations_identifier)   

            activations_identifier = 'scat_transform_J=3_L=4_M=224_N=224_gpool=False_naturalscenes_principal_components=1000_rescaled'
            scores_iden = activations_identifier + '_' + region

            EncodingScore(activations_identifier=activations_identifier,
                           dataset=DATASET,
                           region=region,
                           device= 'cuda').get_scores(scores_iden)
            gc.collect()


