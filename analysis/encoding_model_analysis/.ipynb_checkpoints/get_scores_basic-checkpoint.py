import os 
import sys
ROOT_DIR = os.getenv('MB_ROOT_PATH')
sys.path.append(ROOT_DIR)
DATA_DIR = os.getenv('MB_DATA_PATH')
ACTIVATIONS_PATH = os.path.join(DATA_DIR,'activations')   



import torchvision
from tools.processing import *
from tools.utils import get_activations_iden, get_scores_iden
from analysis.neural_data_regression.tools.extractor import Activations
import xarray as xr
import numpy as np
import torch
from analysis.neural_data_regression.tools.regression import *
from analysis.neural_data_regression.tools.scorer import *
import torchvision
from sklearn.linear_model import Ridge
from models.all_models.alexnet import Alexnet

alexnet_pytorch =  torchvision.models.alexnet(pretrained=True)




DATASET = 'naturalscenes' # define dataset
REGIONS = ['general'] # define resgions (list including any of V1, V2, V3, V4, general)

MAX_POOL = True # whether u have maxpooled activations
MODE = 'cv' #  use cv for 10-fold cross validation




MODEL_DICT = {
    
       'alexnet':{
                'iden':'alexnet', 
                'model':alexnet_pytorch,
                'layers': ['features.12'], 
                'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                'num_layers':5,
                'num_features':256,
                'dim_reduction_type':None,
                'max_pool':MAX_POOL,
                'alphas': [10**i for i in range(3,7)]},  # the range of ridge alphas for regression, use [0] for OLS
}
 


for region in REGIONS:
    
    for model_name, model_info in MODEL_DICT.items():
        print(model_name)
        
        activations_identifier = get_activations_iden(model_info, DATASET, MODE)
        print(activations_identifier)

        # get model activations  
        activations = Activations(model=model_info['model'],
                                layer_names=model_info['layers'],
                                dataset=DATASET,
                                preprocess=model_info['preprocess'],
                                mode = MODE
                            )           
        activations.get_array(ACTIVATIONS_PATH,activations_identifier) 
        
        
        
        
        # get model score
        for alpha in model_info['alphas']: # if using PLS, can comment this out
            
            regression_model = Ridge(alpha=alpha)    # set your regresion type here        
            scores_identifier = get_scores_iden(model_info, activations_identifier, region, DATASET, MODE, alpha)
            
            scorer(model_name=model_info['iden'],
                   activations_identifier=activations_identifier,
                   scores_identifier=scores_identifier,
                   regression_model=regression_model,
                   dataset=DATASET,
                   mode=MODE,
                   regions=[region])
