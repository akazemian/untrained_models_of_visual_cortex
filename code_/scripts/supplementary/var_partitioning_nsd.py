import random
#Get PCs on the test set for both models:
import os
import torch
import random
import pickle
from tqdm import tqdm
import numpy as np
import xarray as xr

from code_.encoding_score.benchmarks.nsd import load_nsd_data, filter_activations
from code_.encoding_score.regression.scores_tools import compute_bootstrap_distribution

from sklearn.linear_model import Ridge
from code_.encoding_score.regression.regression_tools import regression_shared_unshared
from code_.encoding_score.regression.torch_cv import TorchRidgeGCV
from config import CACHE, MAJAJ_DATA, PREDS_PATH
from config import MAJAJ_TRAIN_IDS, MAJAJ_TEST_IDS, TRAIN_IDS_DEMO, TEST_IDS_DEMO
ALPHA_RANGE = [10**i for i in range(-10,10)]

from code_.eigen_analysis.compute_pcs import compute_model_pcs
from code_.model_activations.loading import load_model, load_full_identifier
from code_.eigen_analysis.utils import _PCA
from code_.model_activations.activation_extractor import Activations
from config import setup_logging
import logging

setup_logging()

DATASET = 'naturalscenes' 
SUBJECTS = [0,1,2,3,4,5,6,7]
region = 'ventral visual stream'
FEATURES = 3000 
DEVICE = 'cuda'
N_COMPONENTS = [1000]
TOTAL_COMPONENTS = 1000
MODEL_NAMES = ['expansion', 'alexnet_trained']
BATCH_SIZE = 10
LAYERS = 5
N_BOOTSTRAPS = 1000
n_rows = 872
ALL_SAMPLED_INDICES = np.random.choice(n_rows, (N_BOOTSTRAPS, n_rows), replace=True) 

for MODEL_NAME in MODEL_NAMES:
    compute_model_pcs(model_name=MODEL_NAME, 
                      features=FEATURES, 
                      layers=LAYERS, 
                      batch_size=BATCH_SIZE,
                      dataset=DATASET, 
                      components=TOTAL_COMPONENTS, 
                      device=DEVICE)
    
    # project activations onto the computed PCs 
    for n_components in N_COMPONENTS:
        
        pca_identifier = load_full_identifier(model_name=MODEL_NAME, 
                                                        features=FEATURES, 
                                                        layers=LAYERS, 
                                                        dataset=DATASET,
                                                        principal_components = TOTAL_COMPONENTS)
        
        activations_identifier = load_full_identifier(model_name=MODEL_NAME, 
                                                features=FEATURES, 
                                                layers=LAYERS, 
                                                dataset=DATASET,
                                                principal_components = n_components)            
        
        logging.info(f"Extracting activations and projecting onto the first {n_components} PCs")
        
        #load model
        model = load_model(model_name=MODEL_NAME, 
                           features=FEATURES, 
                               layers=LAYERS,
                               device=DEVICE)
    
        # compute activations and project onto PCs
        Activations(model=model, 
                    dataset=DATASET, 
                    pca_iden = pca_identifier,
                    n_components = n_components, 
                    batch_size = BATCH_SIZE,
                    device= DEVICE).get_array(activations_identifier)  

identifiers = {
    'alexnet_identifier':f'alexnet_conv5_layers=5_features=256_dataset={DATASET}_principal_components=1000',
    'expansion_identifier': f'expansion_features={FEATURES}_layers=5_dataset={DATASET}_principal_components=1000',
    'joint_identifier': f'alexnet_expansion_concat'
}

for name, iden in identifiers.items():
    
    for subject in tqdm(SUBJECTS): 
        
        file_path = os.path.join(PREDS_PATH,f'{iden}_{region}_{subject}.pkl')
        
        
        ids_train, neural_data_train = load_nsd_data(mode ='unshared', subject = subject,
                                                        region = region)
        ids_test, neural_data_test = load_nsd_data(mode ='shared', subject = subject,
                                                    region = region) 
        y_train = neural_data_train['beta'].values
        y_test = neural_data_test['beta'].values
        
        if name == 'joint_identifier':
    
            alexnet_PCs = xr.open_dataarray(os.path.join(CACHE,'activations',
                                            identifiers['alexnet_identifier']), 
                                            engine='netcdf4')  
            alexnet_PCs_train = filter_activations(data = alexnet_PCs, ids = ids_train) 
            alexnet_PCs_test = filter_activations(data = alexnet_PCs, ids = ids_test)
            
            expansion_PCs= xr.open_dataarray(os.path.join(CACHE,'activations',
                                            identifiers['expansion_identifier']),
                                             engine='netcdf4')
            expansion_PCs_train = filter_activations(data = expansion_PCs, ids = ids_train) 
            expansion_PCs_test = filter_activations(data = expansion_PCs, ids = ids_test)
    
            X_train = torch.cat([torch.Tensor(alexnet_PCs_train), 
                                 torch.Tensor(expansion_PCs_train)], dim=1)
            X_test = torch.cat([torch.Tensor(alexnet_PCs_test), 
                                torch.Tensor(expansion_PCs_test)], dim=1)
    
        else:
            activation_PCs = xr.open_dataarray(os.path.join(CACHE,'activations',iden),
                                               engine='netcdf4')
            X_train = filter_activations(data = activation_PCs, ids = ids_train) 
            X_test = filter_activations(data = activation_PCs, ids = ids_test)

            

            
        regression = TorchRidgeGCV(
            alphas=ALPHA_RANGE,
            fit_intercept=True,
            scale_X=False,
            scoring='pearsonr',
            store_cv_values=False,
            alpha_per_target=False,
            device = DEVICE)
        
        regression.fit(X_train, y_train)
        best_alpha = float(regression.alpha_)

        y_true, y_predicted = regression_shared_unshared(x_train=X_train,
                                                     x_test=X_test,
                                                     y_train=y_train,
                                                     y_test=y_test,
                                                     model= Ridge(alpha=best_alpha))
        with open(file_path,'wb') as f:
            pickle.dump(y_predicted, f,  protocol=4)
        del y_true, y_predicted

    
