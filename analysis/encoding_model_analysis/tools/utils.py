import xarray as xr
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys

path = '/home/akazemi3/Desktop/MB_Lab_Project/'
sys.path.append(path)
from analysis.encoding_model_analysis.tools.regression import *

from tools.processing import *
from tools.loading import *

import pandas as pd
import seaborn as sns
import pickle
import tables
import numpy as np


    
def get_activations_iden(model_info, dataset, mode):
    
        model_name = model_info['iden'] 
        
        if model_info['max_pool']:
            model_name = model_name + '_mp' 

        activations_identifier = model_name + '_' + f'{model_info["num_layers"]}_layers' + '_' + f'{model_info["num_features"]}_features' 

        if mode == 'pca':
            return activations_identifier + '_' + dataset + '_' + 'pca'
                
        if mode == 'cv' and dataset == 'naturalscenes':
            return activations_identifier + '_' + dataset + '_' + 'shared'
  
        else:
            return activations_identifier + '_' + dataset 

        
        
        

def get_scores_iden(model_info, activations_identifier, region, dataset, mode, alpha=None):
    
    
    if mode=='ridgecv':
        scores_identifier = activations_identifier + '_' + region + '_' + mode
        
    else:
        scores_identifier = activations_identifier + '_' + region + '_' + mode + '_' + f'ridge(alpha={alpha})' 
    
    return scores_identifier








