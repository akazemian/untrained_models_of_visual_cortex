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
# from model_features.models.expansion_5_layers_change_weights import Expansion5L
from model_features.models.expansion_5_layers import Expansion5L
from model_features.models.expansion_3_layers import Expansion
from model_features.models.fully_connected_5_layers import FullyConnected5L
from model_features.models.expansion_5_layers_linear import Expansion5LLinear
from model_evaluation.predicting_brain_data.benchmarks.nsd import load_nsd_data
#from model_features.models.rainbow_trained import RainbowModelTrained

# define local variables
DATASET = 'naturalscenes'
REGIONS = ['ventral visual stream','early visual stream', 'midventral visual stream','V1-4']

DEVICE = 'cuda' 
GLOBAL_POOL = False
subject = 0

for region in REGIONS:
    
    for num_filters in [30000]:
        
            if subject == -1:
                
                activations_identifier = f'expansion_linear_{num_filters}_dataset={DATASET}_shared_images'
                print(activations_identifier)
                image_ids = load_nsd_data(mode='shared', subject=0, region=region, return_data=False)
                Activations(model=Expansion5LLinear(filters_5=num_filters, gpool = False).Build() ,
                        layer_names=['last'],
                        dataset=DATASET,
                        device= DEVICE,
                        batch_size = 3,
                        compute_mode = 'fast',
                        subject_images=image_ids).get_array(activations_identifier) 
            # else:
                activations_identifier = f'expansion_{num_filters}_dataset={DATASET}_subject={subject}'
                print(activations_identifier)
                image_ids = load_nsd_data(mode='unshared', subject=subject, region=region, return_data=False)
                Activations(model=Expansion5L(filters_5=num_filters, gpool = False).Build() ,
                        layer_names=['last'],
                        dataset=DATASET,
                        device= DEVICE,
                        batch_size = 3,
                        compute_mode = 'fast',
                        subject_images=image_ids).get_array(activations_identifier) 

#             activations_identifier = f'expansion_{num_filters}_dataset={DATASET}_subject={subject}'
            scores_iden = activations_identifier + '_' + region 

            print(scores_iden)
            EncodingScore(activations_identifier=activations_identifier,
                       dataset=DATASET,
                       region=region,
                       subject = subject,
                       device= 'cpu').get_scores(scores_iden)

            gc.collect()
    




        
        

