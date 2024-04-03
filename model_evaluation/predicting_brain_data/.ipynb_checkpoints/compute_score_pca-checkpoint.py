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
import numpy as np
# define local variables

MODELS = ['expansion']
LAYERS = 5


DATASET = 'naturalscenes'
REGIONS = ['ventral visual stream'] #'early visual stream',
FEATURES = [3,30,300,3000]
#FEATURES = [12,12*5,12*50]



# DATASET = 'majajhong'
# REGIONS = ['IT']
# FEATURES = [3,30,300,3000,30000]
#FEATURES = [30000]
#FEATURES = [12,12*5,12*50,12*500]


for region in REGIONS:
    print(region)
    
    for model_name in MODELS:
        
        for features in FEATURES:

            TOTAL_COMPONENTS = 100 if features == 3 else 1000
            N_COMPONENTS = list(np.logspace(0, np.log10(TOTAL_COMPONENTS), num=int(np.log10(TOTAL_COMPONENTS)) + 1, base=10).astype(int))
            
            for n_components in N_COMPONENTS:
                
                    print(model_name, features, n_components)
                
                    activations_identifier = load_iden(model_name=model_name, features=features, layers=LAYERS, dataset=DATASET)            
                    
                    print(activations_identifier)
                    model = load_model(model_name=model_name, features=features, layers=LAYERS)
    
    
                    Activations(model=model,
                            layer_names=['last'],
                            dataset=DATASET,
                            device= 'cuda',
                            hook='pca',
                            pca_iden = activations_identifier + f'_components={TOTAL_COMPONENTS}',
                            n_components=n_components,
                            batch_size = 30).get_array(activations_identifier + f'_principal_components={n_components}') 
    
    
                    EncodingScore(activations_identifier=activations_identifier + f'_principal_components={n_components}',
                               dataset=DATASET,
                               region=region,
                               device= 'cuda').get_scores(iden= activations_identifier + f'_principal_components={n_components}')
    
                    gc.collect()

                





        
        

