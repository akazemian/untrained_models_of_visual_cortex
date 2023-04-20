# getting activations for a specific dataset from a specific model. Output is an xarray with dims: features x presentation (stimulus_id)
# from kymatio.torch import Scattering2D
import xarray as xr
import os 
import sys
import torchvision

ROOT_DIR = os.getenv('MB_ROOT_PATH')
print(ROOT_DIR)
sys.path.append(ROOT_DIR)


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
from models.all_models.model_3L_spca import EngineeredModel3LSPCA
from models.all_models.model_3L_pca import EngineeredModel3LPCA
from models.all_models.model_4L import EngineeredModel4L
from models.all_models.alexnet_untrained_wide import AlexnetU
from models.all_models.alexnet_untrained_wide_spca import AlexnetUSPCA
from models.all_models.alexnet_untrained_wide_pca import AlexnetUPCA

from models.all_models.model_3L_vone import EngineeredModel3LVOne
from models.all_models.model_2L_vone import EngineeredModel2LVOne


torch.manual_seed(0)
torch.cuda.manual_seed(0)
untrained_alexnet = torchvision.models.alexnet(pretrained=False)
 
ROOT = os.getenv('MB_DATA_PATH')
ACTIVATIONS_PATH = os.path.join(ROOT,'activations')   


# define constants
DATASET = 'naturalscenes'
REGIONS = ['V4']

# DATASET = 'majajhong'
# REGIONS = ['V4','IT']

MODE = 'train'
PCA = False
RANDOM_PROJ = False



def get_activations_iden(model_info):
    
        model_name = model_info['iden'] 
        
        if model_info['max_pool']:
            model_name = model_name + '_mp' 

        activations_identifier = model_name + '_' + f'{model_info["num_layers"]}_layers' + '_' + f'{model_info["num_features"]}_features' 

        if model_info['pca']:
            activations_identifier = activations_identifier + '_' + f'{model_info["num_pca_components"]}' + '_'  + f'{model_info["pca_dataset"]}' + '_' + 'pcs' 

        return activations_identifier + '_' + DATASET
  



def get_scores_iden(activations_identifier, region, mode, alpha):
    
    scores_identifier = activations_identifier + '_' + REGION + '_' + MODE + '_' + f'ridge(alpha={alpha})' 
    return scores_identifier






MODEL_DICT = {
                       
#            'model_vone':{
#                 'model':EngineeredModel2LVOne(im_size=96,filters_2=10000).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=96).PreprocessRGB, 
#                 'num_layers':2,
#                 'num_features':10000,
#                 'pca':False,
#                 'max_pool':True,
#                 'alphas':[10**i for i in range(3,7)]},
   

#             'model_vone_224':{
#                 'model':EngineeredModel3LVOne(im_size=224).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#                 'num_layers':3,
#                 'num_features':10000,
#                 'pca':False,
#                 'max_pool':True,
#                 'alphas':[10**i for i in range(3,7)]},     
    
            'model 100 pcs':{
                'iden':'model',
                'model':EngineeredModel3LPCA(n_components=100).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=96).PreprocessGS, 
                'num_layers':3,
                'num_features':10000,
                'pca':True,
                'num_pca_components':100,
                'pca_dataset':'nsd',
                'max_pool':False,
                'alphas': [10**i for i in range(2,4)]},
    
    
            'model 256 pcs':{
                'iden':'model',
                'model':EngineeredModel3LPCA(n_components=256).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=96).PreprocessGS, 
                'num_layers':3,
                'num_features':10000,
                'pca':True,
                'num_pca_components':256,
                'pca_dataset':'nsd',
                'max_pool':False,
                'alphas':[10**i for i in range(2,4)]},
    
    

            'model 1000 pcs':{
                'iden':'model',
                'model':EngineeredModel3LPCA(n_components=1000).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=96).PreprocessGS, 
                'num_layers':3,
                'num_features':10000,
                'pca':True,
                'num_pca_components':1000,
                'pca_dataset':'nsd',
                'max_pool':False,
                'alphas':[10**i for i in range(2,4)]},
                

            'model 5000 pcs':{
                'iden':'model',
                'model':EngineeredModel3LPCA(n_components=5000).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=96).PreprocessGS, 
                'num_layers':3,
                'num_features':10000,
                'pca':True,
                'num_pca_components':5000,
                'pca_dataset':'nsd',
                'max_pool':False,
                'alphas':[10**i for i in range(2,4)]}  
}
 


for region in REGIONS:
    
    for model_name, model_info in MODEL_DICT.items():

        print(model_name)

        activations_identifier = get_activations_iden(model_info)
        print(activations_identifier)

        activations = Activations(model=model_info['model'],
                            layer_names=model_info['layers'],
                            dataset=DATASET,
                            preprocess=model_info['preprocess'],
                            max_pool=model_info['max_pool'],
                            pca = PCA,
                            random_proj = RANDOM_PROJ,
                            mode = MODE
                            )                   
        activations.get_array(ACTIVATIONS_PATH,activations_identifier)   
        
        for alpha in model_info['alphas']:
            
            print('alpha',alpha)
            regression_model = Ridge(alpha=alpha)            
            scores_identifier = activations_identifier + '_' + region + '_' + MODE + '_' + f'ridge(alpha={alpha})' 
            
            scorer(model_name=model_info['iden'],
                   activations_identifier=activations_identifier,
                   scores_identifier=scores_identifier,
                   regression_model=regression_model,
                   dataset=DATASET,
                   mode=MODE,
                   regions=[region]
                  )
