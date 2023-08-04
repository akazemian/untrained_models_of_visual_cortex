
import sys
import xarray as xr
import numpy as np
import torch
import os
from .regression import *
from .scorers.majajhong import *
# from .scorers.nsd_2 import *
from .scorers.nsd import *

import warnings
warnings.filterwarnings('ignore')
from tools.scorers.function_types import Regression

ROOT_DIR = os.getenv('MB_ROOT_PATH')
sys.path.append(ROOT_DIR)
DATA_DIR = os.getenv('MB_DATA_PATH')
MODEL_SCORES_PATH = os.path.join(DATA_DIR,'model_scores_final')
        
        
    
def scorer(model_name: str, 
           activations_identifier: str,
           scores_identifier: str,
           dataset: str,
           mode: str,
           *args,**kwargs) -> None: 
        
        """
    
        Obtain and save the encoding score (unit-wise pearson r values) of a particular model for a particular dataset 


        Parameters
        ----------
        
        model_name:
                Name of model for which the encoding score is being obtained
        
        activations_identifier:
                Name of the file containing the model activations  
        
        scores_identifier:
                Name of the file to save scores in
        
        regression_model:
                Model used for regression
        
        dataset:
                Name of neural dataset (majajhong, naturalscenes)
        
        mode:
                The mode with which regression is carried out (train, test, cv)
        
        """

        if not os.path.exists(MODEL_SCORES_PATH):
                os.mkdir(MODEL_SCORES_PATH)

                
        if os.path.exists(os.path.join(MODEL_SCORES_PATH,activations_identifier,f'{scores_identifier}')):
            print(f'model scores are already saved in {os.path.join(MODEL_SCORES_PATH,activations_identifier)} as {scores_identifier}')


        else:
            print('obtaining model scores...')        

            if dataset == 'naturalscenes':
                
                match mode:
                    
                    case 'train': 
                        ds = nsd_scorer_unshared_cv(model_name,activations_identifier, 
                                                scores_identifier, regression_model,*args,**kwargs)
                    case 'test': 
                        ds = nsd_scorer_all(model_name,activations_identifier, 
                                                 scores_identifier, regression_model,*args,**kwargs)
                    case 'cv': 
                        ds = nsd_scorer_shared_cv(model_name,activations_identifier, 
                                                 scores_identifier, regression_model,*args,**kwargs)
                    
                    case 'ridgecv':
                        ds = nsd_scorer_end_to_end(model_name,activations_identifier, 
                                                 scores_identifier, *args,**kwargs)

                    
            elif dataset == 'majajhong':

                match mode:
                    case 'train': 
                        ds = majajhong_scorer_subset_cv(model_name,activations_identifier, 
                                                   scores_identifier, regression_model,*args,**kwargs)
                    case 'test':
                        ds = majajhong_scorer_all(model_name,activations_identifier, 
                                                   scores_identifier, regression_model,*args,**kwargs)
                    case 'cv':
                        ds = majajhong_scorer_all_cv(model_name,activations_identifier, 
                                                     scores_identifier, regression_model,*args,**kwargs)
                                             
                    case 'ridgecv':
                        ds = majajhong_scorer_end_to_end(model_name,activations_identifier, 
                                                 scores_identifier, *args,**kwargs)
                                                   

            
            if not os.path.exists(os.path.join(MODEL_SCORES_PATH,activations_identifier)):
                os.mkdir(os.path.join(MODEL_SCORES_PATH,activations_identifier))
            ds.to_netcdf(os.path.join(MODEL_SCORES_PATH,activations_identifier,f'{ds.name.values}'))
            print(f'model scores are now saved in {os.path.join(MODEL_SCORES_PATH,activations_identifier)} as {ds.name.values}')
            return
