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
        
        
        # if model_info['dim_reduction_type'] in ['pca','spca']: 
        #     activations_identifier = activations_identifier + '_' + f'{model_info["dim_reduction_type"]}' + '_' + f'{model_info["max_dims"]}' + '_'  + f'{model_info["pca_dataset"]}' + '_' + 'pcs' 

        # if model_info['dim_reduction_type'] == 'rp': 
        #     activations_identifier = activations_identifier + '_' + f'{model_info["dim_reduction_type"]}' + '_' + f'{model_info["n_dims"]}' + '_' + 'rps' 
        
        if mode == 'cv' and dataset == 'naturalscenes':
            return activations_identifier + '_' + dataset + '_' + 'shared'
  
        else:
            return activations_identifier + '_' + dataset 

        
        
        
        


def get_scores_iden(model_info, activations_identifier, region, dataset, mode, alpha=None):
    
#     if model_info['dim_reduction_type'] in ['pca','spca']:
#         activations_identifier = activations_identifier.split(f'_{dataset}')[0] 
#         scores_identifier = activations_identifier + '_' + f'{model_info["n_dims"]}' + '_' + 'subset' + '_' + dataset + '_' + region + '_' + mode + '_' + f'ridge(alpha={alpha})'
            
#     else:
    
    
    if mode=='ridgecv':
        scores_identifier = activations_identifier + '_' + region + '_' + mode
        
    else:
        scores_identifier = activations_identifier + '_' + region + '_' + mode + '_' + f'ridge(alpha={alpha})' 
    
    return scores_identifier








def get_best_alpha(data_dict,alphas,legend_name_dict=None):
    

    ROOT_PATH = '/data/atlas/model_scores'
    df = pd.DataFrame()
    index = 0
    
    for model, regions in data_dict.items():
                
        for a in alphas:
            regression = f'Ridge(alpha={a})' 
                
            for region in regions:

                data = xr.open_dataset(os.path.join(ROOT_PATH,f'{model}_{regression}'))
                r_values = data.where(data.region == region,drop=True).r_value.values
                mean_r = np.mean(r_values)
                df_tmp =  pd.DataFrame({'mean_score':mean_r,
                                        'alpha':a,
                                        'model':model,
                                        'region':region},index=[index])
                df = pd.concat([df,df_tmp])
                index += 1

    
    df = df.groupby(['region','model','alpha']).mean().reset_index()
    if legend_name_dict is not None:
        df.model = df.model.map(legend_name_dict)
    max_idx = df.groupby(['model','region']).agg({'mean_score':('idxmax')}).reset_index()['mean_score'].tolist()
    df_max_alphas = df.loc[max_idx].reset_index(drop=True)

    return df_max_alphas



