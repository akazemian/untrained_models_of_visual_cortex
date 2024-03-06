import os 
import sys
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
import warnings
warnings.filterwarnings('ignore')
print(os.getenv('BONNER_ROOT_PATH'))
from image_tools.processing import *
from model_evaluation.predicting_brain_data.regression.scorer import EncodingScore
from model_features.activation_extractor import Activations
import gc
from model_evaluation.predicting_brain_data.benchmarks.nsd import load_nsd_data
from model_features.models.models import load_model, load_iden

# define local variables
DATASET = 'naturalscenes_shuffled'
REGIONS = ['ventral visual stream']
FEATURES = [3,30,300,3000]

# DATASET = 'majajhong_shuffled'
# REGIONS = ['IT']
# FEATURES = [3,30,300,3000,30000]

MODELS = ['expansion']

FILTERS = [None]
LAYERS = 5


for region in REGIONS:
    
    print(region)
        
    for model_name in MODELS:
        
        print(model_name)
        
        for features in FEATURES:
            
            for random_filters in FILTERS:
    
                
                activations_identifier = load_iden(model_name=model_name, features=features, random_filters = random_filters, layers=LAYERS, dataset=DATASET)
                print(activations_identifier)
                
                model = load_model(model_name=model_name, features=features, random_filters = random_filters, layers=LAYERS)

                Activations(model=model,
                        layer_names=['last'],
                        dataset=DATASET,
                        device= 'cuda',
                        batch_size = 50).get_array(activations_identifier) 


                EncodingScore(activations_identifier=activations_identifier,
                           dataset=DATASET,
                           region=region,
                           device= 'cuda').get_scores(iden= activations_identifier + '_' + region)

                gc.collect()





        
        

