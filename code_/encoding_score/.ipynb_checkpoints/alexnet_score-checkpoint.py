from image_tools.processing import *
from model_evaluation.predicting_brain_data.regression.scorer import EncodingScore
from model_features.activation_extractor import Activations
from model_features.models.utils import load_model, load_iden
from model_evaluation.results.predicting_brain_data.tools import get_bootstrap_data
import gc

# define local variables
# DATASET = 'majajhong'
# REGIONS = ['V4','IT']


DATASET = 'naturalscenes'
REGIONS = ['early visual stream', 'ventral visual stream','midventral visual stream']

MODEL_NAME = '_alexnet'    
DEVICE = 'cpu'


for region in REGIONS:
    
    print(region)
                
    activation_iden_list = []
    
    for layer_num in range(1,6):
        
                
        activations_identifier = load_full_iden(model_name=model_name, features=features, random_filters = random_filters, layers=LAYERS, dataset=DATASET)
        print(activations_identifier)
                
        model = load_model(model_name=model_name, features=features, random_filters = random_filters, layers=LAYERS)
        
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




