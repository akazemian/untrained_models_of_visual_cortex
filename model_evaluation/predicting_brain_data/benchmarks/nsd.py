
from ..regression.regression import regression_shared_unshared, pearson_r
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
import gc

ROOT = os.getenv('BONNER_ROOT_PATH')
sys.path.append(ROOT)
from config import CACHE, NSD_NEURAL_DATA      

SHARED_IDS_PATH = os.path.join(ROOT, 'image_tools','nsd_ids_shared')
SHARED_IDS = pickle.load(open(SHARED_IDS_PATH, 'rb'))
SHARED_IDS = [image_id.strip('.png') for image_id in SHARED_IDS]
ALPHA_RANGE = [10**i for i in range(10)]
    
    
def normalize(X):
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X = np.nan_to_num(X)
    return X





def nsd_scorer(activations_identifier: str, 
               region: str):
    


        activations_data = xr.open_dataarray(os.path.join(CACHE,'activations',activations_identifier), engine='h5netcdf')  
        
        ds = xr.Dataset(data_vars=dict(r_value=(["neuroid"], [])),
                                coords={'x':(['neuroid'], []), 
                                        'y':(['neuroid'], []), 
                                        'z':(['neuroid'], []),
                                        'subject': (['neuroid'], []),
                                        'region': (['neuroid'], [])
                                         })

        for subject in tqdm(range(8)):


            ids_train, neural_data_train, var_name_train = load_nsd_data(mode ='unshared',
                                                        subject = subject,
                                                        region = region)
            X_train = filter_activations(data = activations_data, ids = ids_train)       
            y_train = neural_data_train[var_name_train].values
            
            
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
            
            
            ids_test, neural_data_test, var_name_test = load_nsd_data(mode ='shared',
                                                                      subject = subject,
                                                                      region = region)           
            X_test = filter_activations(data = activations_data, ids = ids_test)                
            y_test = neural_data_test[var_name_test].values                    
            
            y_true, y_predicted = regression_shared_unshared(x_train=X_train,
                                                             x_test=X_test,
                                                             y_train=y_train,
                                                             y_test=y_test,
                                                             model= Ridge(alpha=best_alpha))
            r = pearson_r(y_true,y_predicted)

            ds_tmp = xr.Dataset(data_vars=dict(r_value=(["neuroid"], r)),
                                        coords={'x':(['neuroid'],neural_data_test.x.values ),
                                                'y':(['neuroid'],neural_data_test.y.values ),
                                                'z':(['neuroid'], neural_data_test.z.values),
                                                'subject': (['neuroid'], [subject for i in range(len(r))]),
                                                'region': (['neuroid'], [region for i in range(len(r))])
                                                 })

            ds = xr.concat([ds,ds_tmp],dim='neuroid')   

        ds['name'] = activations_identifier + '_' + region 
        return ds            
            

     
    
def get_best_model_layer(activations_identifier, region):
    
        best_score = 0
        
        for iden in activations_identifier:
            
            print('getting scores for:',iden)
            activations_data = xr.open_dataarray(os.path.join(CACHE,'activations',iden), engine='h5netcdf')  
        
            scores = []
            alphas = [] 
            
            for subject in tqdm(range(8)):

                regression = fit_model_for_subject_roi(subject, region, activations_data)                
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
            
            
            
            
            

def nsd_get_best_layer_scores(activations_identifier: list, region: str):
    
        ds = xr.Dataset(data_vars=dict(r_value=(["neuroid"], [])),
                        coords={'x':(['neuroid'], []), 
                                'y':(['neuroid'], []), 
                                'z':(['neuroid'], []),
                                'subject': (['neuroid'], []),
                                'region': (['neuroid'], [])
                                 })


        best_layer, best_alphas = get_best_model_layer(activations_identifier, region)            
        activations_data = xr.open_dataarray(os.path.join(CACHE,'activations',best_layer), engine='h5netcdf')  
        
        for subject in tqdm(range(8)):

            ids_train, neural_data_train, var_name_train = load_nsd_data(mode ='unshared',
                                                        subject = subject,
                                                        region = region)
            X_train = filter_activations(data = activations_data, ids = ids_train)       
            y_train = neural_data_train[var_name_train].values


            ids_test, neural_data_test, var_name_test = load_nsd_data(mode ='shared',
                                                                      subject = subject,
                                                                      region = region)           

            X_test = filter_activations(data = activations_data, ids = ids_test)               
            y_test = neural_data_test[var_name_test].values                    
            
            y_true, y_predicted = regression_shared_unshared(x_train=X_train,
                                                             x_test=X_test,
                                                             y_train=y_train,
                                                             y_test=y_test,
                                                             model= Ridge(alpha=best_alphas[subject]))
            r = pearson_r(y_true,y_predicted)

            ds_tmp = xr.Dataset(data_vars=dict(r_value=(["neuroid"], r)),
                                        coords={'x':(['neuroid'],neural_data_test.x.values ),
                                                'y':(['neuroid'],neural_data_test.y.values ),
                                                'z':(['neuroid'], neural_data_test.z.values),
                                                'subject': (['neuroid'], [subject for i in range(len(r))]),
                                                'region': (['neuroid'], [region for i in range(len(r))])
                                                 })

            ds = xr.concat([ds,ds_tmp],dim='neuroid')   

        ds['name'] = best_layer + '_' + region + '_' + 'best_layer' 
        return ds       
        
        
            

                
    
    
    
def fit_model_for_subject_roi(subject:int, region:str, activations_data:xr.DataArray()):
            
            ids_train, neural_data_train, var_name_train = load_nsd_data(mode ='unshared',
                                                                         subject = subject,
                                                                         region = region)

            X_train = filter_activations(data = activations_data, ids = ids_train)       
            y_train = neural_data_train[var_name_train].values

            regression = TorchRidgeGCV(
                alphas=ALPHA_RANGE,
                fit_intercept=True,
                scale_X=False,
                scoring='pearsonr',
                store_cv_values=False,
                alpha_per_target=False)
            
            regression.fit(X_train, y_train)
            return regression
            
            
            
            
            
            
    
    
    
            
            
def load_nsd_data(mode: str, subject: int, region: str) -> torch.Tensor:
        
        """
        
        Loads the neural data from disk for a particular subject and region.


        Parameters
        ----------
        mode:
            The type of neural data to load ('shared' or 'unshared')
            
        subject:
            The subject number 
        
        region:
            The region name
            
        return_ids: 
            Whether the image ids are returned 
        

        Returns
        -------
        A Tensor of Neural data, or Tensor of Neural data and stimulus ids
        
        """
        path = os.path.join(NSD_NEURAL_DATA,f'roi={region}/preprocessed/z_score=session.average_across_reps=True/subject={subject}.nc')
        
        var_name = f'allen2021.natural_scenes.preprocessing=fithrf_GLMdenoise_RR.roi={region}.z_score=session.average_across_reps=True.subject={subject}'

        
        ds = xr.open_dataset(path, engine='h5netcdf')

        if mode == 'unshared':
            data = ds.where(~ds.presentation.stimulus_id.isin(SHARED_IDS),drop=True)

        elif mode == 'shared':
            data = ds.where(ds.presentation.stimulus_id.isin(SHARED_IDS),drop=True)
            
        ids = list(data.presentation.stimulus_id.values)
            
        return ids, data, var_name
        
        
            
def filter_activations(data: xr.DataArray, ids: list) -> torch.Tensor:
            
        """
    
        Filters model activations using image ids.


        Parameters
        ----------
        data:
            Model activation data
            
        ids:
            image ids
        

        Returns
        -------
        A Tensor of model activations filtered by image ids
        
        """
        
        data = data.set_index({'presentation':'stimulus_id'})
        activations = data.sel(presentation=ids)
        activations = activations.sortby('presentation', ascending=True)

        return activations.values
            

