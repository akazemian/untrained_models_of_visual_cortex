from ..regression.regression import regression_shared_unshared, pearson_r
from ..regression.regression_cv_mod import RidgeCVMod
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
ALPHA_RANGE = [10**i for i in range(1,10)]

TRAIN_IDS =  pickle.load(open(os.path.join(ROOT,'model_evaluation/predicting_brain_data/benchmarks','majaj_train_ids'), "rb"))
TEST_IDS =  pickle.load(open(os.path.join(ROOT,'model_evaluation/predicting_brain_data/benchmarks','majaj_test_ids'), "rb"))

    
def majajhong_scorer(model_name: str, 
                           activations_identifier: str, 
                           scores_identifier: str, 
                           region: str):
    

        
        ds = xr.Dataset(data_vars=dict(r_value=(["r_values"], [])),
                                coords={'subject': (['r_values'], []),
                                        'region': (['r_values'], [])
                                         })

        
        X_train = load_activations(activations_identifier, mode = 'train')
        X_test = load_activations(activations_identifier, mode = 'test')
        

        pbar = tqdm(total = 2)
        for subject in SUBJECTS:


            y_train = load_majaj_data(subject= subject, region= region, mode = 'train')
            y_test = load_majaj_data(subject= subject, region= region, mode = 'test')


            regression = RidgeCVMod(alphas=ALPHA_RANGE, store_cv_values = False,
                                  alpha_per_target = True, scoring = 'pearson_r')
            regression.fit(X_train, y_train)
            best_alpha = st.mode(regression.alpha_)[0]
            print('best alpha:',best_alpha)


            y_true, y_predicted = regression_shared_unshared(x_train=X_train,
                                                         x_test=X_test,
                                                         y_train=y_train,
                                                         y_test=y_test,
                                                         model= Ridge(alpha=best_alpha))
            r = pearson_r(y_true,y_predicted)

            ds_tmp = xr.Dataset(data_vars=dict(r_value=(["r_values"], r)),
                            coords={'subject': (['r_values'], [subject for i in range(len(r))]),
                                    'region': (['r_values'], [region for i in range(len(r))])
                                     })
                
            ds = xr.concat([ds,ds_tmp],dim='r_values')   
            pbar.update(1)

        ds['name'] = scores_identifier
        return ds           
        
        
        
        
        


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
        
        activations_data = xr.open_dataset(os.path.join(os.path.join(CACHE,'activations',activations_identifier)), engine='netcdf4')
        activations_data = activations_data.set_index({'presentation':'stimulus_id'})

        if mode == 'train':
            activations_data = activations_data.sel(presentation=TRAIN_IDS)
        
        
        elif mode == 'test':            
            activations_data = activations_data.sel(presentation=TEST_IDS)           
        
        activations_data = activations_data.sortby('presentation', ascending=True)
        
        return torch.Tensor(activations_data.x.values)
    
    
    

    
    
    
    
