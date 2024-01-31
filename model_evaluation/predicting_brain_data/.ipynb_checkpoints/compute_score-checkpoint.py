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
# define local variables

# DATASET = 'naturalscenes'
# REGIONS = ['ventral visual stream','midventral visual stream'] #'early visual stream',

DATASET = 'majajhong'
REGIONS = ['IT']


# MODELS = ['ViT_large_embed']#,
MODELS = ['fully_random_1000']
FEATURES = [3000]
LAYERS = 5


for region in REGIONS:
    
    print(region)
        
    for model_name in MODELS:
        
        print(model_name)
        
        for features in FEATURES:
    
                        
                activations_identifier = load_iden(model_name=model_name, features=features, layers=LAYERS, dataset=DATASET)

                print(activations_identifier)

                model = load_model(model_name=model_name, features=features, layers=LAYERS)


                Activations(model=model,
                        layer_names=['last'],
                        dataset=DATASET,
                        device= 'cuda',
                        batch_size = 2).get_array(activations_identifier) 


                EncodingScore(activations_identifier=activations_identifier,
                           dataset=DATASET,
                           region=region,
                           device= 'cpu').get_scores(iden= activations_identifier + '_' + region)

                gc.collect()





        
        

