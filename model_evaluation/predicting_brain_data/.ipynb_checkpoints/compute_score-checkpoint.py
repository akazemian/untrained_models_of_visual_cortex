import os 
import sys
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
import warnings
warnings.filterwarnings('ignore')

from image_tools.processing import *
from model_evaluation.predicting_brain_data.regression.scorer import EncodingScore
from model_features.activation_extractor import Activations
from model_features.models.expansion import Expansion, Expansion2L
import gc
from model_evaluation.predicting_brain_data.benchmarks.nsd import load_nsd_data
from model_features.models.models import load_model, load_iden
from model_features.models.ViT import ViT
# define local variables

DATASET = 'naturalscenes'
REGIONS = ['early visual stream','midventral visual stream']

# DATASET = 'majajhong'
# REGIONS = ['V4']


MODELS = ['expansion'] #'expansion_linear']#,
FEATURES = [3,30,300]
LAYERS = 3



for region in REGIONS:
    
    print(region)
        
    for model_name in MODELS:
        
        print(model_name)
        
        for features in FEATURES:
    
            print(features)
            
            activations_identifier = load_iden(model_name=model_name, features=features, layers=LAYERS, dataset=DATASET)
            activations_identifier = activations_identifier + '_6x6'
            print(activations_identifier)
            model = load_model(model_name=model_name, features=features, layers=LAYERS)
         

            # activations_identifier = 'transformer_untarined_learned_pos_larger_hidden_large_lin_head'
            # model = ViT().Build()

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





        
        

