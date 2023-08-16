
import sys
import xarray as xr
import numpy as np
import torch
import os
from ..benchmarks.nsd import nsd_scorer
from ..benchmarks.majajhong import majajhong_scorer

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
                 activations_identifier: str,
                 dataset: str,
                 region:str):
        
        self.model_name = model_name
        self.activations_identifier = activations_identifier
        self.dataset = dataset
        self.region = region
        
        if not os.path.exists(os.path.join(CACHE,'encoding_scores')):
            os.mkdir(os.path.join(CACHE,'encoding_scores'))


        
    @staticmethod
    def cache_file(scores_identifier):
        return os.path.join('encoding_scores',scores_identifier)

    
    @cache(cache_file)
    def get_scores(self, scores_identifier):       

         
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

                ds = nsd_scorer(scores_identifier = scores_identifier, 
                                model_name = self.model_name,
                                activations_identifier = self.activations_identifier, 
                                region = self.region
                                )


            case 'majajhong':

                ds = majajhong_scorer(scores_identifier = scores_identifier, 
                                model_name = self.model_name,
                                activations_identifier = self.activations_identifier, 
                                region = self.region)
        
        print('model scores are saved in cache')
        return ds
