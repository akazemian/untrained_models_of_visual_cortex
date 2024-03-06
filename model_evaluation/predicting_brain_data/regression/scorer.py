
import sys
import xarray as xr
import numpy as np
import torch
import os
from ..benchmarks.nsd import nsd_scorer, nsd_scorer_subjects, nsd_get_best_layer_scores, nsd_scorer_principal_components
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
                 activations_identifier: str|list,
                 dataset: str,
                 region:str,
                 device:str,
                 subject=None, 
                 n_components = None, 
                 best_layer=False):
        
        self.activations_identifier = activations_identifier
        self.dataset = dataset
        self.region = region
        self.device = device
        self.subject = subject
        self.n_components = n_components
        self.best_layer = best_layer

        
        if not os.path.exists(os.path.join(CACHE,'encoding_scores_torch')):
            os.mkdir(os.path.join(CACHE,'encoding_scores_torch'))


    @staticmethod
    def cache_file(iden):
        return os.path.join('encoding_scores_torch',iden)
    
    
    @cache(cache_file)
    def get_scores(self, iden):       
                     
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
            
            case 'naturalscenes' | 'naturalscenes_shuffled':
                if self.best_layer:
                    ds = nsd_get_best_layer_scores(activations_identifier= self.activations_identifier, 
                                                   region= self.region,
                                                  device = self.device)                  
                else:
                    if self.subject is not None:
                                                
                        ds = nsd_scorer_subjects(activations_identifier = self.activations_identifier, 
                                    region = self.region,
                                    device = self.device,
                                    subject = self.subject)
                        
                        
                    elif self.n_components is not None:
                        
                        ds = nsd_scorer_principal_components(activations_identifier = self.activations_identifier, 
                                    region = self.region,
                                    device = self.device,
                                    n_components = self.n_components)
                    else:                         
                        
                        
                        ds = nsd_scorer(activations_identifier = self.activations_identifier, 
                                    region = self.region,
                                    device = self.device)
                        


                    
            case 'majajhong' | 'majajhong_shuffled':
                
                if self.best_layer:
                    ds = majajhong_get_best_layer_scores(activations_identifier= self.activations_identifier, 
                                                         region= self.region,
                                                         device = self.device)                  
                
                elif self.n_components is not None:
                        
                        ds = majajhong_scorer_principal_components(activations_identifier = self.activations_identifier, 
                                    region = self.region,
                                    device = self.device,
                                    n_components = self.n_components)
                
                else:

                    ds = majajhong_scorer(activations_identifier = self.activations_identifier, 
                                        region = self.region,
                                        device = self.device)        
   

        return ds


    
        
    