
from ..regression import *
from ..regression_cv_mod import RidgeCVMod
from ..scorers.function_types import Regression

import sys
import xarray as xr
import numpy as np
import torch
import os
from tools.loading import get_image_labels
import random 
from tqdm import tqdm 
import pickle 
import warnings
warnings.filterwarnings('ignore')
import scipy
from sklearn.preprocessing import scale
import random    
random.seed(0)
import scipy.stats as st


ROOT = os.getenv('MB_DATA_PATH')
ACTIVATIONS_PATH = os.path.join(ROOT,'activations')
NEURAL_DATA_PATH = os.path.join(ROOT,'neural_data')
DATASET = 'naturalscenes'
PATH_TO_NSD_SHARED_IDS = os.path.join(NEURAL_DATA_PATH,'nsd_shared_ids')
file = open(PATH_TO_NSD_SHARED_IDS, 'rb')
SHARED_IDS = pickle.load(file)




    
    
def normalize(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    
    
    
    
def nsd_scorer_end_to_end(model_name: str, 
                           activations_identifier: str, 
                           scores_identifier: str, 
                           region: str,
                           alpha_values: list):
    


        activations_data = xr.open_dataarray(os.path.join(ACTIVATIONS_PATH,activations_identifier))  
        activations_data.values = normalize(activations_data.values)
        
        ds = xr.Dataset(data_vars=dict(r_value=(["neuroid"], [])),
                                coords={'x':(['neuroid'], []), 
                                        'y':(['neuroid'], []), 
                                        'z':(['neuroid'], []),
                                        'subject': (['neuroid'], []),
                                        'region': (['neuroid'], [])
                                         })


        pbar = tqdm(total = 8)
        for subject in range(8):


            ids_train, neural_data_train, var_name_train = load_nsd_data(mode ='unshared',
                                   subject = subject,
                                   region = region,
                                   return_ids = True)

            X_train = filter_activations(data = activations_data, ids = ids_train)       
            y_train = neural_data_train[var_name_train].values

            regression = RidgeCVMod(alphas=alpha_values, store_cv_values = False,
                                  alpha_per_target = True, scoring = 'pearson_r')
            regression.fit(X_train, y_train)
            best_alpha = st.mode(regression.alpha_)[0]
            print('best alpha:',best_alpha)

            ids_test, neural_data_test, var_name_test = load_nsd_data(mode ='shared',
                                   subject = subject,
                                   region = region,
                                   return_ids = True)           

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
            pbar.update(1)

        ds['name'] = scores_identifier
        return ds            
            

     
    

    
def load_nsd_data(mode: str, subject: int, region: str, return_ids: bool = True) -> torch.Tensor:
        
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
        path = f'/data/shared/for_atlas/roi={region}/preprocessed/z_score=session.average_across_reps=True/subject={subject}.nc'
        
        var_name = f'allen2021.natural_scenes.preprocessing=fithrf_GLMdenoise_RR.roi={region}.z_score=session.average_across_reps=True.subject={subject}'

        
        ds = xr.open_dataset(path)

        if mode == 'unshared':
            data = ds.where(~ds.presentation.stimulus_id.isin(SHARED_IDS),drop=True)

        elif mode == 'shared':
            data = ds.where(ds.presentation.stimulus_id.isin(SHARED_IDS),drop=True)
            print()
            
        ids = list(data.presentation.stimulus_id.values)
            
        
        if return_ids:
            return ids, data, var_name
        
        else:
            return data, var_name
           
        
        
            
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
            

