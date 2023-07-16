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
from tools.utils import get_activations_iden, get_scores_iden
from tools.loading import *
from analysis.encoding_model_analysis.tools.extractor import Activations
from scipy.io import loadmat
import xarray as xr
import numpy as np
import torch
import matplotlib.pyplot as plt
from analysis.encoding_model_analysis.tools.regression import *
from analysis.encoding_model_analysis.tools.scorer import *
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


# define paths

BEST_ALPHA_PATH = '/data/atlas/regression_alphas_final'    
ACTIVATIONS_PATH = '/data/atlas/activations'

# define constants
DATASET = 'naturalscenes'
REGIONS = ['general']
# DATASET = 'majajhong'
# REGIONS = ['V4','IT']

MAX_POOL = True
MODE = 'test'







MODEL_DICT= {
    
    
    'expansion model 3L':{
                'iden':'expansion_model_rgb',
                #'model':ExpansionModel(batches_3 = 10, filters_3=10000).Build(),
                'layers': ['last'], 
                #'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                'num_layers':3,
                'num_features':100000,
                'dim_reduction_type':None,
                'max_pool':MAX_POOL,
                'alphas': 'expansion_model_rgb_mp'},
}


for region in REGIONS:
    print('region:',region)
    

    
    for model_name, model_info in MODEL_DICT.items():
    
        file_name = model_info['alphas']
        file = open(os.path.join(BEST_ALPHA_PATH,f'{file_name}_{DATASET}_{region}'),'rb')
        best_alphas = pickle.load(file)
        file.close()

        alpha = best_alphas[model_name]
        regression_model = Ridge(alpha=alpha)
        print('regression_model:',regression_model)
        
        activations_identifier = get_activations_iden(model_info, DATASET, MODE)
        
        scores_identifier = get_scores_iden(model_info, activations_identifier, region, DATASET, MODE, alpha)        
        scorer(model_name =model_name,
                           activations_identifier=activations_identifier,
                           scores_identifier=scores_identifier,
                           regression_model=regression_model,
                           dataset=DATASET,
                           mode=MODE,
                           regions=[region]),
#            n_dims = None, #model_info['n_dims'],
#            dim_reduction_type = None#'pca'
               
#           )



 

