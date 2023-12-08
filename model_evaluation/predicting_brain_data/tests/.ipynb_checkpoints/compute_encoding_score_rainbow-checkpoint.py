import os 
import sys
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
import warnings
warnings.filterwarnings('ignore')

from image_tools.processing import *
from model_evaluation.predicting_brain_data.regression.scorer import EncodingScore
from model_evaluation.utils import get_st_activations_iden
from model_features.activation_extractor import Activations
import gc
from model_features.models.rainbow import RainbowModel
#from model_features.models.rainbow_trained import RainbowModelTrained

# define local variables
# DATASET = 'majajhong'
# REGIONS = ['V4','IT']

MODEL_NAME = 'scat_transform_pcs'
DATASET = 'naturalscenes'
REGIONS = ['general']
DEVICE = 'cuda' 
GLOBAL_POOL = False

for region in REGIONS:

    activations_identifier = 'rainbow_no_norm_32_1024_10000'               

    Activations(model=RainbowModel(global_pool=GLOBAL_POOL).Build() ,
                layer_names=['last'],
                dataset=DATASET,
                device= DEVICE,
                batch_size = 5,
                compute_mode = 'fast').get_array(activations_identifier)   

    scores_iden = activations_identifier + '_' + region

    EncodingScore(activations_identifier=activations_identifier,
                   dataset=DATASET,
                   region=region,
                   device= 'cpu').get_scores(scores_iden)
    gc.collect()

