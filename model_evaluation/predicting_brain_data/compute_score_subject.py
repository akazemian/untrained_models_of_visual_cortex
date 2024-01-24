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
from model_features.models.ViT import ViT
# define local variables

DATASET = 'naturalscenes'
REGIONS = ['early visual stream','midventral visual stream']
# DATASET = 'majajhong'
# REGIONS = ['V4']


MODELS = ['expansion']#,'fully_connected', 'expansion_linear']
FEATURES = [3000]
LAYERS = 3
DEVICE = 'cuda'
# subject = 0


for subject in range(1,8,1):
    
    
    for region in REGIONS:

        for model_name in MODELS:

            for features in FEATURES:

                model = load_model(model_name=model_name, features=features, layers=LAYERS)
                activations_identifier = load_iden(model_name=model_name, features=features, layers=LAYERS, dataset=DATASET)

                if subject == -1:

                    activations_identifier = activations_identifier + '_shared_images'
                    print(activations_identifier)

                    image_ids = load_nsd_data(mode='shared', subject=0, region=region, return_data=False)
                    Activations(model=model,
                            layer_names=['last'],
                            dataset=DATASET,
                            device= DEVICE,
                            batch_size = 5,
                            subject_images=image_ids).get_array(activations_identifier) 
                else:
                    activations_identifier = activations_identifier + f'_subject={subject}'
                    print(activations_identifier)

                    image_ids = load_nsd_data(mode='unshared', subject=subject, region=region, return_data=False)
                    Activations(model=model,
                            layer_names=['last'],
                            dataset=DATASET,
                            device= DEVICE,
                            batch_size = 5,
                            subject_images=image_ids).get_array(activations_identifier) 

                    scores_iden = activations_identifier + '_' + region 

                    EncodingScore(activations_identifier=activations_identifier,
                               dataset=DATASET,
                               region=region,
                               subject = subject,
                               device= 'cpu').get_scores(scores_iden)

                gc.collect()





        
        

