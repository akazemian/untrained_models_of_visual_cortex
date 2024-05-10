from model_evaluation.predicting_brain_data.regression.scorer import EncodingScore
from model_features.activation_extractor import Activations
from model_features.models.models import load_model, load_full_iden
from model_features.models.configs import analysis_cfg as cfg
from model_evaluation.results.predicting_brain_data.tools import get_bootstrap_data
import gc
import numpy as np


DATASET = 'naturalscenes'
MODEL_NAME= 'expansion'
N_BOOTSTRAPS = 1000
N_ROWS = cfg[DATASET]['test_data_size']
ALL_SAMPLED_INDICES = np.random.choice(N_ROWS, (N_BOOTSTRAPS, N_ROWS), replace=True) # Sample indices for all 



        
for features in cfg[DATASET]['analysis']['pca']['features']:

        TOTAL_COMPONENTS = 100 if features == 3 else 1000
        N_COMPONENTS = list(np.logspace(0, np.log10(TOTAL_COMPONENTS), num=int(np.log10(TOTAL_COMPONENTS)) + 1, base=10).astype(int))
        
        for n_components in N_COMPONENTS:
            
                activations_identifier = load_full_iden(model_name=MODEL_NAME, 
                                                    features=features, 
                                                    layers=cfg[DATASET]['analysis']['pca']['layers'], 
                                                    dataset=DATASET,
                                                    components = n_components)            
                
                pca_identifier = load_full_iden(model_name='expansion', 
                                                    features=features, 
                                                    layers=cfg[DATASET]['analysis']['pca']['layers'], 
                                                    dataset=DATASET,
                                                    components = TOTAL_COMPONENTS), 
                
                print(features, n_components, activations_identifier)
                model = load_model(model_name=MODEL_NAME, 
                                   features=features, 
                                   layers=cfg[DATASET]['analysis']['pca']['layers'], )

                    
                Activations(model=model,
                        layer_names=['last'],
                        dataset=DATASET,
                        device= 'cuda',
                        hook='pca',
                        pca_iden = pca_identifier,
                        n_components=n_components,
                        batch_size = 30).get_array(activations_identifier) 


                EncodingScore(activations_identifier=activations_identifier,
                           dataset=DATASET,
                           region=cfg[DATASET]['regions'],
                           device= 'cuda').get_scores(iden= activations_identifier)

                gc.collect()


get_bootstrap_data(model_name= MODEL_NAME,
                    features=cfg[DATASET]['analysis']['pca']['features'],
                    layers = cfg[DATASET]['analysis']['pca']['layers'],
                    principal_components=[1,10,100,1000],
                    dataset=DATASET, 
                    subjects=cfg[DATASET]['subjects'],
                    file_name = 'pca',
                    region=cfg[DATASET]['regions'],
                    all_sampled_indices=ALL_SAMPLED_INDICES,
                    device='cpu')
gc.collect()

            





    
    

