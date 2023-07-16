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
MODE = 'test'

# changing parameter
FILE_NAME = 'model_3L_mp'
    
MODELS = [   
           f'model_3L_mp_100000_nsd_pca',
        
#             f'alexnet_untrained_wide_mp_10000_nsd_pca',

#             f'alexnet_mp_nsd_pca',
    
    
         ]


for region in REGIONS:
    print('region:',region)
    
    file = open(os.path.join(BEST_ALPHA_PATH,f'{FILE_NAME}_{DATASET}_{region}'),'rb')
    best_alphas = pickle.load(file)
    file.close()
    
    for model_name in MODELS:

        N_COMPONENTS = [10, 100, 256, 1000, 5000, 10000] #if '10000' in model_name else [10, 100, 256]
        max_components = 10000 #if '10000' in model_name else 256


        for n_components in N_COMPONENTS:
            print('n_components:',n_components)
                



            alpha = best_alphas[model_name + '_' + DATASET  + '_' + f'{n_components}']
            regression_model = Ridge(alpha=alpha)
            print('regression_model:',regression_model)

            activations_identifier =  model_name + '_' + DATASET + '_' + f'{max_components}'
            scores_identifier = scores_identifier = model_name + '_' + DATASET + '_' + f'{n_components}' + '_' + region + '_' + MODE + '_' + f'Ridge(alpha={alpha})'

            scorer(model_name=model_name,
                   activations_identifier=activations_identifier,
                   scores_identifier=scores_identifier,
                   regression_model=regression_model,
                   dataset=DATASET,
                   mode=MODE,
                   regions=[region],
                   n_components = n_components
                  )



 

