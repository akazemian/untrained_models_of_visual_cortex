import os 
import sys
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
import warnings
warnings.filterwarnings('ignore')

from image_tools.processing import *
from model_evaluation.predicting_brain_data.regression.scorer import EncodingScore
from model_features.activation_extractor import Activations
from model_features.models.models import load_model, load_iden
import gc

# define local variables
# DATASET = 'majajhong'
# REGIONS = ['V4','IT']


DATASET = 'naturalscenes'
REGIONS = ['early visual stream', 'ventral visual stream','midventral visual stream']

MODEL_NAME = 'alexnet'    
DEVICE = 'cpu'


for region in REGIONS:
    
    print(region)
                
    activation_iden_list = []
    
    for layer_num in range(1,6):
        
        activations_identifier = load_iden(model_name=MODEL_NAME, dataset=DATASET, layers=layer_num)
        activation_iden_list.append(activations_identifier)
        
        model = load_model(model_name=MODEL_NAME, layers=layer_num)
        
        Activations(model=model,
                layer_names=['last'],
                dataset=DATASET,
                device='cpu',
                batch_size=10).get_array(activations_identifier)
    
    
    best_iden = load_iden(model_name=MODEL_NAME, dataset=DATASET, layers='best')
    scores_iden =  best_iden + '_' + region
    
    
    EncodingScore(activations_identifier=activation_iden_list,
                    dataset=DATASET,
                    region=region,
                    device=DEVICE,
                    best_layer=True).get_scores(scores_iden)




