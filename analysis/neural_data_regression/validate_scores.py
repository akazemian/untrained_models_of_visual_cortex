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
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
from sklearn.cross_decomposition import PLSRegression

# define paths

BEST_ALPHA_PATH = '/data/atlas/regression_alphas'    
ACTIVATIONS_PATH = '/data/atlas/activations'

# define constants
DATASET = 'naturalscenes_zscored_processed'
REGIONS = ['V4']
# DATASET = 'majajhong'
# REGIONS = ['V4','IT']


MODE = 'test'

# changing parameter
FILE_NAME = 'model_3L_PCA'
    
MODELS = [   
          # 'alexnet_untrained_mp',
          # 'alexnet_mp'


#           'model_3L_mp_100',              
#           'model_3L_mp_1000',              
#           'model_3L_mp_10000',    
          'model_3L_PCA'  
         ]

for region in REGIONS:
    print('region:',region)
    
    file = open(os.path.join(BEST_ALPHA_PATH,f'{FILE_NAME}_{DATASET}_{region}'),'rb')
    best_alphas = pickle.load(file)
    file.close()
    
    for model_name in MODELS:
    
        alpha = best_alphas[model_name]
        regression_model = Ridge(alpha=alpha)
        print('regression_model:',regression_model)
        
        activations_identifier =  model_name + '_' + DATASET
        scores_identifier = activations_identifier + '_' + region + '_' + MODE + '_' + f'Ridge(alpha={alpha})' 
        scorer(model_name=model_name,
               activations_identifier=activations_identifier,
               scores_identifier=scores_identifier,
               regression_model=regression_model,
               dataset=DATASET,
               mode=MODE,
               regions=[region],
              )



 

