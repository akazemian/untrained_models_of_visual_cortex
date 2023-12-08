import os 
import sys
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
import warnings
warnings.filterwarnings('ignore')

from image_tools.processing import *
from model_evaluation.predicting_brain_data.regression.scorer import EncodingScore
from model_evaluation.utils import get_activations_iden, get_best_layer_iden
from model_features.activation_extractor import Activations
from model_features.models.models import load_model_dict
import gc

# define local variables

DATASET = 'majajhong'
REGIONS = ['V4','IT']

# DATASET = 'naturalscenes'
# REGIONS = ['general']#'V1','V2','V3','V4']

DEVICE = 'cuda' 
GLOBAL_POOL = False

model_name = 'alexnet'
models = ['alexnet_conv1','alexnet_conv2','alexnet_conv3','alexnet_conv4','alexnet_conv5']
    
# model_name = 'alexnet_untrained'
# models = ['alexnet_untrained_conv1',
#           'alexnet_untrained_conv2','alexnet_untrained_conv3','alexnet_untrained_conv4',
#           'alexnet_untrained_conv5']     
        
idens = []

for model_name in models:
    
    print('model: ',model_name)
    model_info = load_model_dict(model_name, gpool=GLOBAL_POOL)
    model_info['hook'] = None
    
    activations_identifier = get_activations_iden(model_info, DATASET) 
    idens.append(activations_identifier)    
    
    for region in REGIONS:

        Activations(model=model_info['model'],
                    layer_names=model_info['layers'],
                    dataset=DATASET,
                    hook = model_info['hook'],
                    device= DEVICE,
                    batch_size = 50,
                    compute_mode = 'fast').get_array(activations_identifier)   


        
for region in REGIONS:
    
        scores_iden = get_best_layer_iden(model_name, DATASET, region, GLOBAL_POOL)
        
        encoding_score = EncodingScore(model_name=model_info['iden'],
                        activations_identifier=idens,
                        dataset=DATASET,
                        region=region,
                        best_layer=True).get_scores(scores_iden)
                        
        gc.collect()


