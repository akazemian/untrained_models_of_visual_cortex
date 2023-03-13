
import sys
import xarray as xr
import numpy as np
import torch
import os
from .regression import *
from .scorers.majajhong import *
from .scorers.nsd import *

sys.path.append("/home/akazemi3/MB_Lab_Project") 
import warnings
warnings.filterwarnings('ignore')
from tools.scorers.function_types import Regression


MODEL_SCORES_PATH = '/data/atlas/model_scores'



        
        
        
        
    
def scorer(model_name: str, 
           activations_identifier: str,
           scores_identifier: str,
           regression_model: Regression,
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



        if os.path.exists(os.path.join(MODEL_SCORES_PATH,f'{scores_identifier}')):
            print(f'model scores are already saved in {MODEL_SCORES_PATH} as {scores_identifier}')


        else:
            print('obtaining model scores...')        


            if dataset == 'naturalscenes_zscored_processed':
                
                if mode == 'train': 
                    ds = nsd_scorer_unshared_cv(model_name,activations_identifier, 
                                                scores_identifier, regression_model,*args,**kwargs)
                elif mode == 'test': 
                    ds = nsd_scorer_all(model_name,activations_identifier, 
                                                 scores_identifier, regression_model,*args,**kwargs)
                elif mode == 'cv': 
                    ds = nsd_scorer_shared_cv(model_name,activations_identifier, 
                                                 scores_identifier, regression_model,*args,**kwargs)

            elif dataset == 'majajhong':

                if mode == 'train': 
                    ds = majajhong_scorer_subset_cv(model_name,activations_identifier, 
                                                   scores_identifier, regression_model,*args,**kwargs)
                elif mode == 'test':
                    ds = majajhong_scorer_all(model_name,activations_identifier, 
                                                   scores_identifier, regression_model,*args,**kwargs)
                elif mode == 'cv':
                    ds = majajhong_scorer_all_cv(model_name,activations_identifier, 
                                                   scores_identifier, regression_model,*args,**kwargs)

            ds.to_netcdf(os.path.join(MODEL_SCORES_PATH,f'{ds.name.values}'))
            print(f'model scores are now saved in {MODEL_SCORES_PATH} as {ds.name.values}')
            return
