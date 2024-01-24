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
from model_features.models.expansion_4_layers import Expansion4L
from model_features.models.expansion_5_layers import Expansion5L
from model_features.models.fully_connected_5_layers import FullyConnected5L
from model_features.models.expansion_5_layers_linear import Expansion5LLinear
from model_features.models.expansion_3_layers import Expansion
from model_evaluation.predicting_brain_data.benchmarks.nsd import load_nsd_data
from model_features.models.models import load_model, load_iden

# define local variables

MODELS = ['expansion', 'expansion_linear', 'fully_connected']
FEATURES = [3000]
DATASET = 'naturalscenes'
REGIONS = ['V1-4','ventral visual stream','early visual stream', 'midventral visual stream']
# DATASET = 'majajhong'
# REGIONS = ['V4','IT']


for region in REGIONS:
    
    print(region)
        
    for model_name in MODELS:
        
        print(model_name)
        
        for features in FEATURES:
    
            print(features)
            
            activations_identifier = load_iden(model_name=model_name, features=features, dataset=DATASET)
            model = load_model(model_name=model_name, features=features)

            Activations(model=model,
                    layer_names=['last'],
                    dataset=DATASET,
                    device= 'cuda',
                    batch_size = 40).get_array(activations_identifier) 


            EncodingScore(activations_identifier=activations_identifier,
                       dataset=DATASET,
                       region=region,
                       device= 'cpu').get_scores(iden= activations_identifier + '_' + region)

            gc.collect()





        
        

