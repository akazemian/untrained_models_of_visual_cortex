import random
#Get PCs on the test set for both models:
import os
import torch
import random
import pickle
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import Ridge

from code_.encoding_score.benchmarks.majajhong import load_majaj_data, load_activations
from code_.encoding_score.regression.regression_tools import regression_shared_unshared
from code_.encoding_score.regression.torch_cv import TorchRidgeGCV
from config import PREDS_PATH
from code_.eigen_analysis.compute_pcs import compute_model_pcs
from code_.model_activations.loading import load_model, load_full_identifier
from code_.eigen_analysis.utils import _PCA
from code_.model_activations.activation_extractor import Activations
import logging
from code_.tools.utils import timeit, setup_logging

setup_logging()

@timeit
def main():

    ALPHA_RANGE = [10**i for i in range(-10,10)]
    DATASET = 'majajhong_demo'
    SUBJECTS = ['Tito','Chabo']
    region = 'IT'
    FEATURES = [3, None]
    DEVICE = 'cuda'
    TOTAL_COMPONENTS = n_components = 100
    MODEL_NAMES = ['expansion', 'alexnet_trained']
    BATCH_SIZE = 10
    LAYERS = 5

    for i, MODEL_NAME in enumerate(MODEL_NAMES):
        compute_model_pcs(model_name=MODEL_NAME, 
                        features=FEATURES[i], 
                        layers=LAYERS, 
                        batch_size=BATCH_SIZE,
                        dataset=DATASET, 
                        components=TOTAL_COMPONENTS, 
                        device=DEVICE)
        
        # project activations onto the computed PCs 
        pca_identifier = load_full_identifier(model_name=MODEL_NAME, 
                                                        features=FEATURES[i], 
                                                        layers=LAYERS, 
                                                        dataset=DATASET,
                                                        principal_components = TOTAL_COMPONENTS)
        
        activations_identifier = load_full_identifier(model_name=MODEL_NAME, 
                                                features=FEATURES[i], 
                                                layers=LAYERS, 
                                                dataset=DATASET,
                                                principal_components = n_components)            
        
        logging.info(f"Extracting activations and projecting onto the first {n_components} PCs")
        
        #load model
        model = load_model(model_name=MODEL_NAME, 
                        features=FEATURES[i], 
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
        'alexnet_identifier':f'alexnet_trained_features=None_layers=5_dataset={DATASET}_principal_components=100',
        'expansion_identifier': f'expansion_features={FEATURES[0]}_layers=5_dataset={DATASET}_principal_components=100',
        'joint_identifier': f'alexnet_expansion_joint'
    }

    for name, iden in identifiers.items():
        
        if name == 'joint_identifier':
            
            alexnet_PCs_train = load_activations(
            activations_identifier=identifiers['alexnet_identifier'],mode='train_demo')
            expansion_PCs_train = load_activations(
                activations_identifier=identifiers['expansion_identifier'],mode='train_demo')

            alexnet_PCs_test = load_activations(
                activations_identifier=identifiers['alexnet_identifier'],mode='test_demo')
            expansion_PCs_test = load_activations(
                activations_identifier=identifiers['expansion_identifier'],mode='test_demo')
            
            X_train = torch.cat([alexnet_PCs_train, expansion_PCs_train], dim=1)
            X_test = torch.cat([alexnet_PCs_test, expansion_PCs_test], dim=1)
        
        else:
            X_train = load_activations(activations_identifier=iden, mode='train_demo')
            X_test = load_activations(activations_identifier=iden, mode='test_demo')


        for subject in tqdm(SUBJECTS):  
            file_path = os.path.join(PREDS_PATH,f'{iden}_{region}_{subject}.pkl')
                        
            y_train = load_majaj_data(subject, region, 'train_demo')
            y_test = load_majaj_data(subject, region, 'test_demo')

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

        
if __name__ == "__main__":
    main()

