
import sys
import xarray as xr
import numpy as np
import torch
import os
from ..benchmarks.nsd import nsd_scorer, nsd_get_best_layer_scores, nsd_scorer_nc
from ..benchmarks.majajhong import majajhong_scorer, majajhong_get_best_layer_scores, majajhong_scorer_nc
from ..benchmarks.things import things_scorer, things_get_best_layer_scores
from config import CACHE       
import functools
import gc
from itertools import permutations



class EncodingScore():
    def __init__(self,
                 activations_identifier: str|list,
                 dataset: str,
                 region:str,
                 device:str,
                 nc:bool=False,
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
        self.nc = nc

        
        if not os.path.exists(os.path.join(CACHE,'encoding_scores_torch')):
            os.mkdir(os.path.join(CACHE,'encoding_scores_torch'))

    
    def get_scores(self, iden=None):       
                     
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
                    nsd_get_best_layer_scores(activations_identifier= self.activations_identifier, 
                                                   region= self.region,
                                                  device = self.device)                  
                elif self.nc:
                    nsd_scorer_nc(activations_identifier= self.activations_identifier, 
                                        region= self.region, 
                                        device = self.device)    
                else:
                    nsd_scorer(activations_identifier = self.activations_identifier, 
                                    region = self.region,
                                    device = self.device)
                        


                    
            case 'majajhong' | 'majajhong_shuffled':
                
                if self.best_layer:
                    majajhong_get_best_layer_scores(activations_identifier= self.activations_identifier, 
                                                    region= self.region,
                                                    device = self.device,
                                                    demo=False)    
                elif self.nc:
                    majajhong_scorer_nc(activations_identifier= self.activations_identifier, 
                                        region= self.region, 
                                        device = self.device)   
                else:
                    majajhong_scorer(activations_identifier = self.activations_identifier, 
                                        region = self.region,
                                        device = self.device,
                                        demo=False)          
                                    
            case 'majajhong_demo' | 'majajhong_demo_shuffled':

                if self.best_layer:
                    majajhong_get_best_layer_scores(activations_identifier= self.activations_identifier, 
                                                    region= self.region,
                                                    device = self.device,
                                                    demo=True)    
 
                else:
                    majajhong_scorer(activations_identifier = self.activations_identifier, 
                                        region = self.region,
                                        device = self.device,
                                        demo=True)   

            case 'things':
                
                if self.best_layer:
                    things_get_best_layer_scores(activations_identifier= self.activations_identifier, 
                                                    region= self.region,
                                                    device = self.device)                  
                
                else:

                    things_scorer(activations_identifier = self.activations_identifier, 
                                        region = self.region,
                                        device = self.device)     
            case _:
                raise ValueError("dataset should be one of 'majajhong', 'naturalscenes' or 'things'")
        return 


    
        
    