import torch
from code_.encoding_score.preprocessing.tools_nsd_data import average_betas_across_reps, z_score_betas_within_sessions
from sklearn.linear_model import Ridge
import xarray as xr
import torch
import os
import random 
from tqdm import tqdm 
import pickle 
import warnings
warnings.filterwarnings('ignore')
import random    
random.seed(0)
from pathlib import Path

from code_.encoding_score.regression.regression_tools import regression_shared_unshared
from code_.encoding_score.regression.torch_cv import TorchRidgeGCV
from code_.model_activations.loading import get_best_layer_path
from config import THINGS_TRAIN_IDS, THINGS_TEST_IDS
from config import CACHE, PREDS_PATH


def load_activations(activations_identifier: str, mode: bool = None) -> torch.Tensor:
            
        """
        Loads model activations.

        Parameters
        ----------
            
        activations_identifier:
            model activations identifier
            
        mode:
            The type of neural data to load ('all', 'train' or 'test')
        
        Returns
        -------
        A Tensor of activation data
        """
        print(CACHE, activations_identifier)
        activations_data = xr.open_dataarray(os.path.join(CACHE,'activations',activations_identifier), engine='netcdf4')
        activations_data = activations_data.set_index({'presentation':'stimulus_id'})

        match mode:
            case 'all':
                pass
            case 'train':
                activations_data = activations_data.sel(presentation=THINGS_TRAIN_IDS)
            case 'test':            
                activations_data = activations_data.sel(presentation=THINGS_TEST_IDS)             
            case _:
                raise ValueError("mode should be one of 'all', 'train' or 'test'")
        activations_data = activations_data.sortby('presentation', ascending=True)
        return torch.Tensor(activations_data.values)   


def things_scorer(activations_identifier: str, 
                       region: str,
                       device:str):

        X_train = load_activations(activations_identifier, mode = 'train')
        X_test = load_activations(activations_identifier, mode = 'test')
                
        for subject in tqdm(range(3)):

            file_path = Path(PREDS_PATH) / f'{activations_identifier}_{region}_{subject}.pkl'
            
            if not file_path.exists():
                print('saving ',file_path)
                
                y_train = load_things_data(subject= subject, region= region, mode = 'train')
                y_test = load_things_data(subject= subject, region= region, mode = 'test')
    
    
                regression = TorchRidgeGCV(
                    alphas=[10**i for i in range(-10, 10)],
                    fit_intercept=True,
                    scale_X=False,
                    scoring='pearsonr',
                    store_cv_values=False,
                    alpha_per_target=False,
                    device = device)
                
                regression.fit(X_train, y_train)
                best_alpha = float(regression.alpha_)
                print('best alpha:',best_alpha)
    
    
                y_true, y_predicted = regression_shared_unshared(x_train=X_train,
                                                             x_test=X_test,
                                                             y_train=y_train,
                                                             y_test=y_test,
                                                             model= Ridge(alpha=best_alpha))
                
                with open(file_path, 'wb') as file:
                    pickle.dump(y_predicted, file)
            
            else:
                print(file_path, ' exists')
        return      

def load_things_data(subject: str, region: str, mode: bool = None) -> torch.Tensor:
        
        """
        Loads the neural data from disk for a particular subject and region.

        Parameters
        ----------
            
        subject:
            The subject number 
        
        region:
            The region name
            
        mode:
            The type of neural data to load ('all', 'train' or 'test')

        Returns
        -------
        A Tensor of Neural data
        """
        
        import sys
        bonner_lib = '/home/akazemi3/Desktop/bonner-libraries/src'
        neural_dim = '/home/akazemi3/Desktop/neural-dimensionality/src'
        root = '/home/akazemi3/Desktop/untrained_models_of_visual_cortex'
        import numpy as np

        sys.path.append(bonner_lib)
        sys.path.append(neural_dim)
        from lib.datasets import things
        neural_data = things.load_dataset(subject=subject, roi=region)
        neural_data = average_betas_across_reps(z_score_betas_within_sessions(neural_data)).to_dataset()
        
        match mode:            
            case 'all':
                pass
            case 'train':
                neural_data = neural_data.where(neural_data.stimulus.isin(THINGS_TRAIN_IDS),drop=True)
            case 'test':
                neural_data = neural_data.where(neural_data.stimulus.isin(THINGS_TEST_IDS),drop=True)
            case _:
                raise ValueError("mode should be one of 'all', 'train' or 'test'")
            
        neural_data = neural_data.sortby('stimulus', ascending=True)
        neural_data = torch.Tensor(neural_data[f'hebart2023.things-data.subject={subject}.roi={region}'].values.squeeze())
        return neural_data




def get_best_model_layer(activations_identifier, region, device):
    
        t = 0
        for iden in activations_identifier:
            
            print('getting scores for:',iden)
            activations_data = load_activations(activations_identifier = iden, mode = 'train') 
            
        
            scores = []
            alphas = [] 
            
            for subject in tqdm(range(3)):

                regression = fit_model_for_subject_roi(subject, region, activations_data, device)                
                scores.append(regression.score_.mean())
                alphas.append(float(regression.alpha_))
            
            mean_layer_score = sum(scores)/len(scores)
            print(mean_layer_score)
            
            if t == 0:
                best_score = mean_layer_score 
                t += 1
            
        
            if  mean_layer_score >= best_score:
                best_score = mean_layer_score
                best_layer = iden
                best_alphas = alphas
            print('best_layer:', best_layer)
            print('best_alphas:', best_alphas)
            
        
        return best_layer, best_alphas
    
    
    
def things_get_best_layer_scores(activations_identifier: list, region: str, device:str):

        best_layer, best_alphas = get_best_model_layer(activations_identifier, region, device)            
        
        X_train = load_activations(best_layer, mode = 'train')
        X_test = load_activations(best_layer, mode = 'test')

        pbar = tqdm(total = 2)
        
        for subject in tqdm(range(3)):
            
            print('best_layer:', best_layer)
            file_path = get_best_layer_path(best_layer=best_layer, dataset='things', 
                                            region=region, subject=subject)

            if not file_path.exists():
                print('saving ',file_path)
                
                y_train = load_things_data(subject= subject, region= region, mode = 'train')
                y_test = load_things_data(subject= subject, region= region, mode = 'test')
    
                y_true, y_predicted = regression_shared_unshared(x_train=X_train,
                                                                 x_test=X_test,
                                                                 y_train=y_train,
                                                                 y_test=y_test,
                                                                 model= Ridge(alpha=best_alphas[subject]))
               
    
                with open(file_path, 'wb') as file:
                        pickle.dump(y_predicted, file)
            else:
                print(file_path, ' exists')
            
            pbar.update(1)

        return       
        
    
    
def fit_model_for_subject_roi(subject:int, region:str, activations_data:xr.DataArray, device:str):
            
            X_train = activations_data
            y_train = load_things_data(subject, region, 'train')

            regression = TorchRidgeGCV(
                alphas=[10**i for i in range(-10, 10)],
                fit_intercept=True,
                scale_X=False,
                scoring='pearsonr',
                store_cv_values=False,
                alpha_per_target=False,
                device=device)
            
            regression.fit(X_train, y_train)
            return regression
                 

