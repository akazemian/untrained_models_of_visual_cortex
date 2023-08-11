import os 
import sys
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
import warnings
warnings.filterwarnings('ignore')

from image_tools.processing import *
from model_evaluation.predicting_brain_data.regression.scorer import EncodingScore
from model_evaluation.utils import get_activations_iden, get_scores_iden
from model_features.activation_extractor import Activations

from model_features.models.expansion_3_layers import Expansion


# define local variables
DATASET = 'naturalscenes'
REGIONS = ['V1']
MAX_POOL = True
MODE = 'ridgecv'
HOOK = None
DEVICE = 'cuda' 
    
model_info = {
                'iden':'expansion_model_test',
                'model':Expansion(filters_3=10000).Build(),
                'layers': ['last'], 
                'num_layers':3,
                'num_features':10000,
}       
 


for region in REGIONS:
        
    activations_identifier = get_activations_iden(model_info, DATASET, MODE)
    activations = Activations(model=model_info['model'],
                            layer_names=model_info['layers'],
                            dataset=DATASET,
                            mode = MODE,
                            hook = HOOK,
                            device=DEVICE,
                            batch_size = 80
                        ).get_array(activations_identifier)   

    scores_identifier = get_scores_iden(model_info, activations_identifier, region, DATASET, MODE)
    
    EncodingScore(model_name=model_info['iden'],
                   activations_identifier=activations_identifier,
                   dataset=DATASET,
                   region=region).get_scores(scores_identifier=scores_identifier)


