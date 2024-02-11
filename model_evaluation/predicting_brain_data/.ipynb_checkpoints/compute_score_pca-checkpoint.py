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

DATASET = 'naturalscenes'
REGIONS = ['ventral visual stream'] #'early visual stream',

# DATASET = 'majajhong'
# REGIONS = ['IT']


# MODELS = ['ViT_large_embed']#,
MODELS = ['expansion']
FEATURES = [3000]
N_COMPONENTS = [1,10,100,10000]  
#FILTERS = [30,300]
LAYERS = 5


for region in REGIONS:
    
    print(region)
        
    for model_name in MODELS:
        
        print(model_name)
        
        for features in FEATURES:

            for n_components in N_COMPONENTS:
            
            #for random_filters in FILTERS:
    
                        
                activations_identifier = load_iden(model_name=model_name, features=features, layers=LAYERS, dataset=DATASET)            
                
                model = load_model(model_name=model_name, features=features, layers=LAYERS)


                Activations(model=model,
                        layer_names=['last'],
                        dataset=DATASET,
                        device= 'cuda',
                        hook='pca',
                        pca_iden = activations_identifier + f'_components={n_components}',
                        n_components=n_components,
                        batch_size = 80).get_array(activations_identifier + f'_principal_components={n_components}') 


                EncodingScore(activations_identifier=activations_identifier,
                           dataset=DATASET,
                           region=region,
                           device= 'cuda').get_scores(iden= activations_identifier + f'_principal_components={n_components}' + '_' + region)

                gc.collect()





        
        

