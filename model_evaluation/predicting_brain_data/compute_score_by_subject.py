import os 
import sys
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
import warnings
warnings.filterwarnings('ignore')

from image_tools.processing import *
from model_evaluation.predicting_brain_data.regression.scorer import EncodingScore
from model_features.activation_extractor import Activations
import gc
from model_evaluation.predicting_brain_data.benchmarks.nsd import load_nsd_data
from model_features.models.models import load_model, load_iden


# define local variables

# DATASET = 'naturalscenes'
# REGIONS = ['midventral visual stream','early visual stream', 'ventral visual stream']
DATASET = 'majajhong'
REGIONS = ['V4','IT']


# model parameters

MODELS = ['ViT']
# MODELS = ['expansion','expansion_linear','fully_connected']
FEATURES = [12*5, 12*50, 12*500]
LAYERS = 5


DEVICE = 'cuda'
subject =  4 # subject to extract activations for

BATCH_SIZE = 2
    
    

    
for region in REGIONS:

    for model_name in MODELS:

        for features in FEATURES:

            # for subject in range(8):
                
                model = load_model(model_name=model_name, features=features, layers=LAYERS)
                activations_identifier = load_iden(model_name=model_name, features=features, layers=LAYERS, dataset=DATASET)

                print(activations_identifier)

    
                if subject == -1:

                    Activations(model=model,
                            dataset=DATASET,
                            device= DEVICE,
                            batch_size = BATCH_SIZE,
                            subject_images=load_nsd_data(mode='shared', subject=0, region=region, return_data=False)
                               ).get_array(activations_identifier + '_shared_images') 
                else:
    
                    Activations(model=model,
                            dataset=DATASET,
                            device= DEVICE,
                            batch_size = BATCH_SIZE,
                            subject_images=load_nsd_data(mode='unshared', subject=subject, region=region, return_data=False)
                               ).get_array(activations_identifier + f'_subject={subject}') 
    
    
                    # EncodingScore(activations_identifier=activations_identifier,
                    #            dataset=DATASET,
                    #            region=region,
                    #            subject = subject,
                    #            device= 'cpu').get_scores(iden = activations_identifier + f'_subject={subject}' + '_' + region )
    
                    # gc.collect()





        
        

