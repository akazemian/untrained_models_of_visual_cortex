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
REGIONS = ['V1']#,'V2','V3','V4','general']
MAX_POOL = True
MODE = 'ridgecv'
HOOK = None
DEVICE = 'cuda' 
    
model_dict={
    
    'expansion 10': {
                'iden':'expansion_model',
                'model':Expansion(filters_3=10).Build(),
                'layers': ['last'], 
                'num_layers':3,
                'num_features':10
    },
    
    
    'expansion 100': {
                'iden':'expansion_model',
                'model':Expansion(filters_3=100).Build(),
                'layers': ['last'], 
                'num_layers':3,
                'num_features':100
    },
    'expansion 1000': {
                'iden':'expansion_model',
                'model':Expansion(filters_3=1000).Build(),
                'layers': ['last'], 
                'num_layers':3,
                'num_features':1000
    },
    'expansion 10000': {
                'iden':'expansion_model',
                'model':Expansion(filters_3=10000).Build(),
                'layers': ['last'], 
                'num_layers':3,
                'num_features':10000
    }
}       
 


for _, model_info in model_dict.items():
    
    for region in REGIONS:
        
        activations_identifier = get_activations_iden(model_info, DATASET, MODE)

        Activations(model=model_info['model'],
                    layer_names=model_info['layers'],
                    dataset=DATASET,
                    mode = MODE,
                    hook = HOOK,
                    device= DEVICE,
                    batch_size = 80,
                    compute_mode='slow').get_array(activations_identifier)   

        scores_identifier = get_scores_iden(model_info, activations_identifier, region, DATASET, MODE)

        EncodingScore(model_name=model_info['iden'],
                       activations_identifier=activations_identifier,
                       dataset=DATASET,
                       region=region).get_scores(scores_identifier=scores_identifier)


