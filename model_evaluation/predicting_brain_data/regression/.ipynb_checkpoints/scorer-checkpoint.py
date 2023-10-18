
import sys
import xarray as xr
import numpy as np
import torch
import os
from ..benchmarks.nsd import nsd_scorer, nsd_get_best_layer_scores
from ..benchmarks.majajhong import majajhong_scorer, majajhong_get_best_layer_scores, majajhong_scorer_cv

import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
from config import CACHE       
import functools
import gc
    
def cache(file_name_func):

    def decorator(func):
        
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):

            file_name = file_name_func(*args, **kwargs) 
            cache_path = os.path.join(CACHE, file_name)
            
            if os.path.exists(cache_path):
                return 
            
            result = func(self, *args, **kwargs)
            result.to_netcdf(cache_path, engine='h5netcdf')
            gc.collect()
            return 

        return wrapper
    return decorator



class EncodingScore():
    def __init__(self,
                 model_name: str, 
                 activations_identifier: str|list,
                 dataset: str,
                 region:str,
                 best_layer:bool=False):
        
        self.model_name = model_name
        self.activations_identifier = activations_identifier
        self.dataset = dataset
        self.region = region
        self.best_layer = best_layer
        
        if not os.path.exists(os.path.join(CACHE,'encoding_scores_torch')):
            os.mkdir(os.path.join(CACHE,'encoding_scores_torch'))


        
    @staticmethod
    def cache_file(ds, scores_identifier):
        return os.path.join('encoding_scores_torch',scores_identifier)

    
        
    @cache(cache_file)
    def save_scores(self, ds, scores_identifier):       
        return ds     
    
    
    def get_scores(self):
        
         
        """
    
        Obtain and save the encoding score (unit-wise pearson r values) of a particular model for a particular dataset 


        Parameters
        ----------
        
        model_name:
                Name of model for which the encoding score is being obtained
        
        activations_identifier:
                Name of the file containing the model activations  
        
        dataset:
                Name of neural dataset (majajhong, naturalscenes)
        
        """

        print('obtaining model scores...')        

        match self.dataset:
            
            case 'naturalscenes':

                if self.best_layer:
                    ds = nsd_get_best_layer_scores(activations_identifier= self.activations_identifier, region= self.region)                  
                else:
                    ds = nsd_scorer(activations_identifier = self.activations_identifier, region = self.region)

                    
            case 'majajhong':
                
                if self.best_layer:
                    ds = majajhong_get_best_layer_scores(activations_identifier= self.activations_identifier, region= self.region)                  
                
                else:

                    ds = majajhong_scorer(model_name = self.model_name,
                                        activations_identifier = self.activations_identifier, 
                                        region = self.region)
        
        scores_identifier = str(ds.name.values)
        return self.save_scores(ds=ds, scores_identifier=scores_identifier)
        


    
        
    