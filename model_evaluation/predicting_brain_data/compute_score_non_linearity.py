import os 
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
from model_features.models.expansion import ExpansionNoWeightShare
from model_features.models.expansion import Expansion5L

# define local variables

# DATASET = 'naturalscenes'
# REGIONS = ['ventral visual stream']

DATASET = 'majajhong'
REGIONS = ['IT']

NL = ['gelu','elu','abs','leaky_relu']

for region in REGIONS:
    
    print(region)
                        
    for nl in NL:
                
                        
                activations_identifier = load_iden(model_name='expansion', features=3000, random_filters = None, layers=5, dataset=DATASET)
                activations_identifier = activations_identifier + '_' + nl
                print(activations_identifier)
                
                model = Expansion5L(filters_5 = 3000, non_linearity=nl).Build()


                Activations(model=model,
                        layer_names=['last'],
                        dataset=DATASET,
                        device= 'cuda',
                        batch_size = 100).get_array(activations_identifier) 


                EncodingScore(activations_identifier=activations_identifier,
                           dataset=DATASET,
                           region=region,
                           device= 'cuda').get_scores(iden= activations_identifier + '_' + region)

                gc.collect()





        
        

