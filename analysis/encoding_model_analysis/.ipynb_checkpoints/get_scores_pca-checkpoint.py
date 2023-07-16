# getting activations for a specific dataset from a specific model. Output is an xarray with dims: features x presentation (stimulus_id)
# from kymatio.torch import Scattering2D
import xarray as xr
import os 
import sys
import torchvision

PATH = '/home/akazemi3/Desktop/MB_Lab_Project/'
sys.path.append(PATH)
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
import random
from sklearn.linear_model import Ridge
from models.all_models.model_2L import EngineeredModel2L
from models.all_models.model_3L import EngineeredModel3L
from models.all_models.alexnet_untrained_wide import AlexnetU
from models.all_models.alexnet_untrained_wide_pca import AlexnetUPCA    

# define paths
ACTIVATIONS_PATH = '/data/atlas/activations'

# define constants
DATASET = 'naturalscenes_zscored_processed'
REGIONS = ['V4']

# DATASET = 'majajhong'
# REGIONS = ['IT']

MODE = 'train'
MAX_POOL = True
PCA = True
RANDOM_PROJ = False

ALPHAS = [10**i for i in range(1,5)]

    
MODEL_DICT = {

            
            #   'model_3L_mp_100000_nsd_pca':{'model':EngineeredModel3L(filters_3=20000,batches_3=5).Build(),
            #   'layers': ['last'], 'preprocess':PreprocessGS},  
    
    
    
#               'model_3L_mp_10000_nsd_pca':{'model':EngineeredModel3L(filters_3=10000).Build(),
#               'layers': ['last'], 'preprocess':PreprocessGS}, 
    
    
#     'alexnet_untrained_wide_mp_10000_nsd_pca':{'model':AlexnetU1(filters_5 = 10000).Build(),
#               'layers': ['mp5'], 'preprocess':PreprocessRGB},
              

 
    
#     'alexnet_mp_nsd_pca':{'model': torchvision.models.alexnet(pretrained=True),
#               'layers': ['features.12'], 'preprocess':PreprocessRGB},
                

}
 

for alpha in ALPHAS:
        

    print('alpha',alpha)
    regression_model = Ridge(alpha=alpha)


    for region in REGIONS:
        for model_name, model_info in MODEL_DICT.items():

        
            N_COMPONENTS = [10, 100, 256, 1000, 5000, 10000] #if '50000' in model_name else [10, 100, 256]
            max_components = 10000 #if '50000' in model_name else 256

            
            for n_components in N_COMPONENTS:
                print('n_components:',n_components)

                for layer in model_info['layers']:
                    print('layer',layer)





                    activations_identifier = model_name + '_' + DATASET  + '_' + f'{max_components}'
                    activations = Activations(model=model_info['model'],
                                        layer_names=[layer],
                                        dataset=DATASET,
                                        preprocess=model_info['preprocess'],
                                        max_pool=MAX_POOL,
                                        pca = PCA,
                                        random_proj = RANDOM_PROJ,
                                        mode = MODE,
                                        n_components = max_components

                                        )                   
                    activations.get_array(ACTIVATIONS_PATH,activations_identifier)     




                    scores_identifier = model_name + '_' + DATASET + '_' + f'{n_components}' + '_' + region + '_' + MODE + '_' + f'Ridge(alpha={alpha})' 
                    scorer(model_name=model_name,
                           activations_identifier=activations_identifier,
                           scores_identifier=scores_identifier,
                           regression_model=regression_model,
                           dataset=DATASET,
                           mode=MODE,
                           regions=[region],
                           n_components = n_components
                          )
