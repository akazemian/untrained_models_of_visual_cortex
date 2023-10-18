from ..regression.regression import regression_shared_unshared, pearson_r, regression_cv
from ..regression.torch_cv import TorchRidgeGCV
from sklearn.linear_model import Ridge

import sys
import xarray as xr
import numpy as np
import torch
import os
import random 
from tqdm import tqdm 
import pickle 
import warnings
warnings.filterwarnings('ignore')
import random    
random.seed(0)
import scipy.stats as st


ROOT = os.getenv('BONNER_ROOT_PATH')
sys.path.append(ROOT)
from config import CACHE, MAJAJ_DATA   
DATASET = 'majajhong'
SUBJECTS = ['Chabo','Tito']
ALPHA_RANGE = [10**i for i in range(10)]

TRAIN_IDS =  pickle.load(open(os.path.join(ROOT,'model_evaluation/predicting_brain_data/benchmarks','majaj_train_ids'), "rb"))
TEST_IDS =  pickle.load(open(os.path.join(ROOT,'model_evaluation/predicting_brain_data/benchmarks','majaj_test_ids'), "rb"))

    
    
    
    
def majajhong_scorer_cv(model_name: str, 
                       activations_identifier: str, 
                       region: str):

        
        ds = xr.Dataset(data_vars=dict(r_value=(["r_values"], [])),
                                coords={'subject': (['r_values'], []),
                                        'region': (['r_values'], [])
                                         })

        
        X = load_activations(activations_identifier)
        
        pbar = tqdm(total = 2)
        for subject in tqdm(SUBJECTS):

            y = load_majaj_data(subject= subject, region= region)

            y_true, y_predicted = regression_cv(x=X, y=y, model = Ridge(alpha=1))

            r = torch.stack(
                [
                    pearson_r(y_true_, y_predicted_)
                    for y_true_, y_predicted_ in zip(y_true, y_predicted)
                ]
            ).mean(dim=0)

            ds_tmp = xr.Dataset(data_vars=dict(r_value=(["r_values"], r)),
                            coords={'subject': (['r_values'], [subject for i in range(len(r))]),
                                    'region': (['r_values'], [region for i in range(len(r))])
                                     })
                
            ds = xr.concat([ds,ds_tmp],dim='r_values')   
            pbar.update(1)

        ds['name'] = activations_identifier + '_' + region + '_cv'
        return ds           
        
        
        

        
        
        
def majajhong_scorer(model_name: str, 
                       activations_identifier: str, 
                       region: str):

        
        ds = xr.Dataset(data_vars=dict(r_value=(["r_values"], [])),
                                coords={'subject': (['r_values'], []),
                                        'region': (['r_values'], [])
                                         })

        
        X_train = load_activations(activations_identifier, mode = 'train')
        X_test = load_activations(activations_identifier, mode = 'test')
        
        pbar = tqdm(total = 2)
        for subject in tqdm(SUBJECTS):

            y_train = load_majaj_data(subject= subject, region= region, mode = 'train')
            y_test = load_majaj_data(subject= subject, region= region, mode = 'test')


            regression = TorchRidgeGCV(
                alphas=ALPHA_RANGE,
                fit_intercept=True,
                scale_X=False,
                scoring='pearsonr',
                store_cv_values=False,
                alpha_per_target=False)
            
            regression.fit(X_train, y_train)
            best_alpha = float(regression.alpha_)
            print('best alpha:',best_alpha)


            y_true, y_predicted = regression_shared_unshared(x_train=X_train,
                                                         x_test=X_test,
                                                         y_train=y_train,
                                                         y_test=y_test,
                                                         model= Ridge(alpha=best_alpha))
            r = pearson_r(y_true,y_predicted)
            print(r.mean())

            ds_tmp = xr.Dataset(data_vars=dict(r_value=(["r_values"], r)),
                            coords={'subject': (['r_values'], [subject for i in range(len(r))]),
                                    'region': (['r_values'], [region for i in range(len(r))])
                                     })
                
            ds = xr.concat([ds,ds_tmp],dim='r_values')   
            pbar.update(1)

        ds['name'] = activations_identifier + '_' + region 
        return ds           
        
        
        
        
        
    
def get_best_model_layer(activations_identifier, region):
    
        
        t = 0
        for iden in activations_identifier:
            
            print('getting scores for:',iden)
            activations_data = load_activations(activations_identifier = iden, mode = 'train') 
            
        
            scores = []
            alphas = [] 
            
            for subject in tqdm(range(len(SUBJECTS))):

                regression = fit_model_for_subject_roi(SUBJECTS[subject], region, activations_data)                
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
    
    

    
    
    
def majajhong_get_best_layer_scores(activations_identifier: list, region: str):
    
        ds = xr.Dataset(data_vars=dict(r_value=(["r_values"], [])),
                                coords={'subject': (['r_values'], []),
                                        'region': (['r_values'], [])
                                         })


        best_layer, best_alphas = get_best_model_layer(activations_identifier, region)            
        
        X_train = load_activations(best_layer, mode = 'train')
        X_test = load_activations(best_layer, mode = 'test')

        for subject in tqdm(range(len(SUBJECTS))):
            
            y_train = load_majaj_data(subject= SUBJECTS[subject], region= region, mode = 'train')
            y_test = load_majaj_data(subject= SUBJECTS[subject], region= region, mode = 'test')

            y_true, y_predicted = regression_shared_unshared(x_train=X_train,
                                                             x_test=X_test,
                                                             y_train=y_train,
                                                             y_test=y_test,
                                                             model= Ridge(alpha=best_alphas[subject]))
            r = pearson_r(y_true,y_predicted)

           
            ds_tmp = xr.Dataset(data_vars=dict(r_value=(["r_values"], r)),
                            coords={'subject': (['r_values'], [SUBJECTS[subject] for i in range(len(r))]),
                                    'region': (['r_values'], [region for i in range(len(r))])
                                     })
                
            ds = xr.concat([ds,ds_tmp],dim='r_values')   
        

        ds['name'] = best_layer + '_' + region + '_' + 'best_layer'
        return ds       
        
        
            

                
    
    
    
def fit_model_for_subject_roi(subject:int, region:str, activations_data:xr.DataArray()):
            
            X_train = activations_data
            y_train = load_majaj_data(subject, region, 'train')

            regression = TorchRidgeGCV(
                alphas=ALPHA_RANGE,
                fit_intercept=True,
                scale_X=False,
                scoring='pearsonr',
                store_cv_values=False,
                alpha_per_target=False)
            
            regression.fit(X_train, y_train)
            return regression
            
            

            
            
def load_majaj_data(subject: str, region: str, mode: bool = None) -> torch.Tensor:
        
        """
        
        Loads the neural data from disk for a particular subject and region.


        Parameters
        ----------
            
        subject:
            The subject number 
        
        region:
            The region name
            
        mode:
            The type of neural data to load ('train' or 'test')
        

        Returns
        -------
        A Tensor of Neural data
        
        """
        
        file_name = f'SUBJECT_{subject}_REGION_{region}'
        file_path = os.path.join(MAJAJ_DATA,file_name)
        neural_data = xr.open_dataset(file_path, engine='netcdf4')

        
        if mode == 'train':

            neural_data = neural_data.where(neural_data.stimulus_id.isin(TRAIN_IDS),drop=True)
        
        
        elif mode == 'test':
            neural_data = neural_data.where(neural_data.stimulus_id.isin(TEST_IDS),drop=True)
            
        
        neural_data = neural_data.sortby('stimulus_id', ascending=True)
        neural_data = torch.Tensor(neural_data['x'].values.squeeze())
        
        return neural_data

            
           
        
            
def load_activations(activations_identifier: str, mode: bool = None) -> torch.Tensor:
            
        """
    
        Loads model activations.


        Parameters
        ----------
            
        activations_identifier:
            model activations identifier
        
            
        mode:
            The type of neural data to load ('train' or 'test')
        

        Returns
        -------
        A Tensor of activation data
        
        """
        
        activations_data = xr.open_dataarray(os.path.join(CACHE,'activations',activations_identifier), engine='netcdf4')
        activations_data = activations_data.set_index({'presentation':'stimulus_id'})

        if mode == 'train':
            activations_data = activations_data.sel(presentation=TRAIN_IDS)
        
        
        elif mode == 'test':            
            activations_data = activations_data.sel(presentation=TEST_IDS)           
        
        
        activations_data = activations_data.sortby('presentation', ascending=True)
        return torch.Tensor(activations_data.values)
    
    
    

    
    
    
    
