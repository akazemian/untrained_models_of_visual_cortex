# getting activations for a specific dataset from a specific model. Output is an xarray with dims: features x presentation (stimulus_id)
# from kymatio.torch import Scattering2D
import xarray as xr
import os 
import sys
import torchvision

path = '/home/akazemi3/Desktop/MB_Lab_Project/'
sys.path.append(path)
# from models.kymatio import *

from tools.processing import *
from models.call_model import *
from tools.loading import *
from analysis.neural_data_regression.tools.extractor import Activations
from scipy.io import loadmat
import xarray as xr
import numpy as np
import torch
import matplotlib.pyplot as plt
from analysis.neural_data_regression.tools.regression import *
from analysis.neural_data_regression.tools.scorer import *
from models.call_model import EngineeredModel
import torchvision
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
from sklearn.cross_decomposition import PLSRegression


    
#torch.manual_seed(0)

#dataset = 'naturalscenes'
dataset = 'naturalscenes_zscored_processed'
regions = ['V1','V2','V3','V4']
# dataset = 'majajhong'
# regions = ['V4','IT']
layers = ['last']
preprocess = PreprocessGS
      
    
model_list = {
    'model_final_mp_10':EngineeredModel(filters_2=10).Build(),
    'model_final_mp_100':EngineeredModel(filters_2=100).Build(),
    'model_final_mp_1000':EngineeredModel(filters_2=1000).Build(),
    'model_final_mp_10000':EngineeredModel(filters_2=10000).Build(),
    'model_final_mp_100000':EngineeredModel(filters_2=10000,batches_2=5).Build()
}


df_max_alphas = pd.read_csv(f'/home/akazemi3/Desktop/MB_Lab_Project/analysis/neural_data_regression/max_alphas_{dataset}_mp.csv')
dict_max_alphas = df_max_alphas.set_index(['region','model']).to_dict()['alpha']



for region in regions:
    print('region',region)
    
    for f in range(1,6):
        
        features = 10**f
        print('features',features)
        model_name = f'model_final_mp_{features}'
        model = model_list[model_name]
        
        
        alpha = dict_max_alphas[(region, features)]
        regression_model = Ridge(alpha=alpha)
        print('regression_model',regression_model)


        # get activations  
        activations = Activations(model=model,
                            layer_names=layers,
                            dataset=dataset,
                            preprocess=preprocess
                            )                  

        activations_identifier = model_name + '_' + dataset
        activations.get_array(activations_path,activations_identifier)     

        
        scores_identifier = activations_identifier + f'_{region}' + f'_Ridge(alpha={alpha})'
        scorer(model_name=model_name,
               activations_identifier=activations_identifier,
               scores_identifier=scores_identifier,
               regression_model=regression_model,
               dataset=dataset,
               mode='test',
               regions=[region]
              )


 

