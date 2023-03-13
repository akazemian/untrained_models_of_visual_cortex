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

    
# define paths
ACTIVATIONS_PATH = '/data/atlas/activations'

# define constants
DATASET = 'naturalscenes_zscored_processed'
REGIONS = ['V4']
                   
MODEL_DICT = {'model_final_mp_all':{'model':EngineeredModel2L().Build(),
              'layers': ['c2'],
              'preprocess':PreprocessGS}}
# MODEL_DICT = {'alexnet_journal':{'model':torchvision.models.alexnet(pretrained=True),
#               'layers': ['features.12'],
#               'preprocess':PreprocessRGB}}

MODE = 'train'
SAVE_BETAS=True
ALPHAS = [0] + [10**i for i in range(11)] 


for a in ALPHAS:
    
    print('alpha',a)
    regression_model = Ridge(alpha=a)


    for model_name, model_info in MODEL_DICT.items():

        
        for layer in model_info['layers']:
            print('layer',layer)
        

        
            activations_identifier = model_name + '_' + DATASET
            activations = Activations(model=model_info['model'],
                                layer_names=[layer],
                                dataset=DATASET,
                                preprocess=model_info['preprocess'],
                                max_pool=True
                                )                   
            activations.get_array(ACTIVATIONS_PATH,activations_identifier)     

    
    
            
            scores_identifier = activations_identifier + '_'  + f'Ridge(alpha={a})' 
            scorer(model_name=model_name,
                   activations_identifier=activations_identifier,
                   scores_identifier=scores_identifier,
                   regression_model=regression_model,
                   dataset=DATASET,
                   mode=MODE,
                   regions=REGIONS,
                   save_betas=SAVE_BETAS
                  )
