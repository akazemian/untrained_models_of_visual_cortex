import xarray as xr
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys

path = '/home/akazemi3/Desktop/MB_Lab_Project/'
sys.path.append(path)
from analysis.neural_data_regression.tools.regression import *

from tools.processing import *
from tools.loading import *

import pandas as pd
import seaborn as sns
import pickle
import tables
import numpy as np





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



