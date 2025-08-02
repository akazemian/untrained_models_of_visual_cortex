
from sklearn.linear_model import Ridge
from dotenv import load_dotenv

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
import gc
from pathlib import Path
from itertools import permutations
from sklearn.model_selection import train_test_split
import numpy as np

from config import PREDS_PATH
from config import ALPHA_RANGE, NSD_NEURAL_DATA
from code_.model_activations.loading import get_best_layer_path
from code_.encoding_score.regression.regression_tools import regression_shared_unshared
from code_.encoding_score.regression.torch_cv import TorchRidgeGCV

load_dotenv()
warnings.filterwarnings('ignore')
random.seed(0)

CACHE = os.getenv("CACHE")


def nsd_scorer(activations_identifier: str, 
               region: str,
              device: str):
        
        for subject in tqdm(range(8)):

            file_path = os.path.join(PREDS_PATH,f'{activations_identifier}_{region}_{subject}.pkl')
            if not os.path.exists(file_path):
            
                activations_data = xr.open_dataarray(os.path.join(CACHE,'activations',activations_identifier), engine='netcdf4')  
                ids_train, neural_data_train = load_nsd_data(mode ='unshared',
                                                    subject = subject,
                                                    region = region)
                X_train = filter_activations(data = activations_data, ids = ids_train)  
                y_train = neural_data_train['beta'].values
            
                regression = TorchRidgeGCV(
                    alphas=ALPHA_RANGE,
                    fit_intercept=True,
                    scale_X=False,
                    scoring='pearsonr',
                    store_cv_values=False,
                    alpha_per_target=False,
                    device=device)
                
                regression.fit(X_train, y_train)
                best_alpha = float(regression.alpha_)
            
            
                ids_test, neural_data_test = load_nsd_data(mode ='shared',
                                                        subject = subject,
                                                        region = region)           
                X_test = filter_activations(data = activations_data, ids = ids_test)                

                y_test = neural_data_test['beta'].values                   
                
                y_true, y_predicted = regression_shared_unshared(x_train=X_train,
                                                                x_test=X_test,
                                                                y_train=y_train,
                                                                y_test=y_test,
                                                                model= Ridge(alpha=best_alpha))
                with open(file_path,'wb') as f:
                    pickle.dump(y_predicted, f,  protocol=4)
                del y_true, y_predicted
            else:
                pass
        return         


def nsd_scorer_nc(activations_identifier:str, region: str, device:str):

    SUBJECTS = [i for i in range(8)]
    # print(permutations(SUBJECTS, 2))
    for subject_y in SUBJECTS:
                                            
        _, data_y = load_nsd_data(mode ='shared', subject = subject_y, region = region)              
        y = data_y['beta'].values  
        
        all_X_subjects = []
        for subject_X in SUBJECTS:
             if subject_X == subject_y:
                  pass
             else:
                _, data_X = load_nsd_data(mode ='shared', subject = subject_X, region = region)
                all_X_subjects.append(data_X['beta'].values)
        X = np.concatenate(all_X_subjects, axis=1)
        
        print('X shape', X.shape)
        print('y shape', y.shape)

        # Perform Z-score normalization across features
        X_zscore = (X - X.mean(axis=0)) / X.std(axis=0)
        y_zscore = (y - y.mean(axis=0)) / y.std(axis=0)
        X_train, X_test, y_train, y_test = train_test_split(X_zscore, y_zscore, test_size=0.2, random_state=42)

        file_path = Path(PREDS_PATH) / f'{activations_identifier}_{region}_{subject_y}.pkl'
        if not file_path.exists():
            print('saving ',file_path)
            
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



def get_best_model_layer(activations_identifier, region, device):
    
        best_score = 0
        
        for iden in activations_identifier:
            
            print('getting scores for:',iden)
            activations_data = xr.open_dataarray(os.path.join(CACHE,'activations',iden), engine='netcdf4')  
        
            scores = []
            alphas = [] 
            
            for subject in tqdm(range(8)):

                regression = fit_model_for_subject_roi(subject, region, activations_data, device)                
                scores.append(regression.score_.mean())
                alphas.append(float(regression.alpha_))
            
            mean_layer_score = sum(scores)/len(scores)
            
            if  mean_layer_score > best_score:
                best_score = mean_layer_score
                best_layer = iden
                best_alphas = alphas
            print('best_layer:', best_layer, 'best_alphas:', best_alphas)
            
            gc.collect()            
        
        return best_layer, best_alphas
    
            

def nsd_get_best_layer_scores(activations_identifier: list, region: str, device:str):

        best_layer, best_alphas = get_best_model_layer(activations_identifier, region, device)            
        
        for subject in tqdm(range(8)):

            print('best_layer:', best_layer)
            file_path = get_best_layer_path(best_layer=best_layer, dataset='naturalscenes', 
                                region=region, subject=subject)
            
            if not os.path.exists(file_path):
                activations_data = xr.open_dataarray(os.path.join(CACHE,'activations',best_layer), engine='netcdf4')  
                ids_train, neural_data_train = load_nsd_data(mode ='unshared',
                                                    subject = subject,
                                                    region = region)
                X_train = filter_activations(data = activations_data, ids = ids_train)       
                y_train = neural_data_train['beta'].values

                ids_test, neural_data_test = load_nsd_data(mode ='shared',
                                                        subject = subject,
                                                        region = region)           

                X_test = filter_activations(data = activations_data, ids = ids_test)               
                y_test = neural_data_test['beta'].values                    
                
                _, y_predicted = regression_shared_unshared(x_train=X_train,
                                                                x_test=X_test,
                                                                y_train=y_train,
                                                                y_test=y_test,
                                                                model= Ridge(alpha=best_alphas[subject]))
                with open(file_path,'wb') as f:
                    pickle.dump(y_predicted, f,  protocol=4)
                del _, y_predicted
            else:
                pass
        return
        
    
    
def fit_model_for_subject_roi(subject:int, region:str, activations_data, device:str):
            
            ids_train, neural_data_train = load_nsd_data(mode ='unshared',
                                                         subject = subject,
                                                         region = region)

            X_train = filter_activations(data = activations_data, ids = ids_train)       
            y_train = neural_data_train['beta'].values

            regression = TorchRidgeGCV(
                alphas=ALPHA_RANGE,
                fit_intercept=True,
                scale_X=False,
                scoring='pearsonr',
                store_cv_values=False,
                alpha_per_target=False,
                device=device)
            
            regression.fit(X_train, y_train)
            return regression    
     
                    
            
            
def load_nsd_data(mode: str, subject: int, region: str, return_data=True) -> torch.Tensor:
        
        SHARED_IDS = pickle.load(open(os.path.join(NSD_NEURAL_DATA, 'nsd_ids_shared'), 'rb'))
        SHARED_IDS = [image_id.strip('.png') for image_id in SHARED_IDS]
        ds = xr.open_dataset(os.path.join(NSD_NEURAL_DATA,region,'preprocessed',f'subject={subject}.nc'),engine='netcdf4')
        
        if mode == 'unshared':
                mask = ~ds.presentation.stimulus.isin(SHARED_IDS)
                ds = ds.sel(presentation=ds['presentation'][mask])

        elif mode == 'shared':
                mask = ds.presentation.stimulus.isin(SHARED_IDS)
                ds = ds.sel(presentation=ds['presentation'][mask])
            
        ids = list(ds.presentation.stimulus.values.astype(str))
            
        if return_data:
            return ids, ds
        else:
            return ids
        
        
            
def filter_activations(data: xr.DataArray, ids: list) -> torch.Tensor:
        
        data = data.where(data['stimulus_id'].isin(ids), drop=True)
        data = data.sortby('presentation', ascending=True)

        return data.values
            
        
        
        

