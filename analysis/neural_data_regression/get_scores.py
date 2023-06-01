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
from tools.utils import get_activations_iden, get_scores_iden

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
from models.all_models.model_3L_spca import EngModel3LSPCA
from models.all_models.model_3L_pca import EngModel3LPCA
from models.all_models.alexnet_untrained_wide import AlexnetUWide
from models.all_models.alexnet_untrained_wide_spca import AlexnetUSPCA
from models.all_models.alexnet_untrained_wide_pca import AlexnetUPCA
from models.all_models.alexnet_pca import AlexnetPCA


from models.all_models.model_3L_abs import EngModel3LAbs
from models.all_models.model_3L_abs_spca import EngModel3LAbsSPCA
from models.all_models.model_3L_abs_pca import EngModel3LAbsPCA
from models.all_models.alexnet import Alexnet
from models.all_models.alexnet_u import AlexnetU
from models.all_models.model_3L_abs_blurpool import EngModel3LAbsBP
from models.all_models.model_3L_abs_blurpool_avgpool import EngModel3LAbsBPAP
from models.all_models.model_3L_abs_blurpool_avgpool_pca import EngModel3LAbsBPAPPCA
from models.all_models.scat_transform import ScatTransform
from models.all_models.scat_transform_kymatio import ScatTransformKymatio


torch.manual_seed(0)
torch.cuda.manual_seed(0)
untrained_alexnet = torchvision.models.alexnet(pretrained=False)
alexnet_pytorch =  torchvision.models.alexnet(pretrained=True)

ROOT = os.getenv('MB_DATA_PATH')
ACTIVATIONS_PATH = os.path.join(ROOT,'activations')   


# define constants
# DATASET = 'naturalscenes'
# REGIONS = ['general']

DATASET = 'naturalscenes'
REGIONS = ['general']

MAX_POOL = True
MODE = 'train'
RANDOM_PROJ = False


# 'model name':{
# 'iden':str,
# 'model':nn.Module,
# 'layers': int, 
# 'preprocess':'function', 
# 'num_layers':int,
# 'num_features':int,
# 'dim_reduction_type':str,
# 'n_dims':int,
# 'max_dims':int,
# 'pca_dataset':str,
# 'max_pool':bool,
# 'alphas': list}






MODEL_DICT = {

#         'scattering transform kymatio':{
#                 'iden':'scat_transform_kymatio_J3_L4_rgb',
#                 'model':ScatTransformKymatio(J = 3, L = 4, M = 32, N = 32, flatten = True, global_mp= False).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=32).PreprocessRGB, 
#                 'num_layers':2,
#                 'num_features':'x',
#                 'dim_reduction_type':None,
#                 'max_pool':False,
#                 'alphas': [0] + [10**i for i in range(3)]},  
    

#         'scattering transform kymatio':{
#                 'iden':'scat_transform_kymatio_J3_L4',
#                 'model':ScatTransformKymatio(J = 3, L = 4, M = 32, N = 32, flatten = True, global_mp= True).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=32).PreprocessGS, 
#                 'num_layers':2,
#                 'num_features':'x',
#                 'dim_reduction_type':False,
#                 'max_pool':True,
#                 'alphas': [0] + [10**i for i in range(3)]},      
    
#     'model abs 3x3 bp 224 ap 10 filters':{
#                 'iden':'model_abs_3x3_bp_224_ap',
#                 'model':EngModel3LAbsBP(filters_3 = 10, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessGS, 
#                 'num_layers':3,
#                 'num_features':10,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas': [10**i for i in range(1,5)]},  
    
#     'model abs 3x3 bp 224 ap 100 filters':{
#                 'iden':'model_abs_3x3_bp_224_ap',
#                 'model':EngModel3LAbsBPAP(filters_3 = 100, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessGS, 
#                 'num_layers':3,
#                 'num_features':100,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas': [10**i for i in range(1,5)]},  
    
#     'model abs 3x3 bp 224 ap 1000 filters':{
#                 'iden':'model_abs_3x3_bp_224_ap',
#                 'model':EngModel3LAbsBPAP(filters_3 = 1000, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessGS, 
#                 'num_layers':3,
#                 'num_features':1000,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas': [10**i for i in range(1,5)]},  
    
#     'model abs 3x3 bp 224 ap 10000 filters':{
#                 'iden':'model_abs_3x3_bp_224_ap',
#                 'model':EngModel3LAbsBPAP(filters_3 = 10000, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessGS, 
#                 'num_layers':3,
#                 'num_features':10000,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas': [10**i for i in range(1,5)]},  
#     'model abs 3x3 bp 224 10 filters':{
#                 'iden':'model_abs_3x3_bp_224',
#                 'model':EngModel3LAbsBP(filters_3 = 10, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessGS, 
#                 'num_layers':3,
#                 'num_features':10,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas': [10**i for i in range(1,5)]},  
    
#     'model abs 3x3 bp 224 100 filters':{
#                 'iden':'model_abs_3x3_bp_224',
#                 'model':EngModel3LAbsBP(filters_3 = 100, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessGS, 
#                 'num_layers':3,
#                 'num_features':100,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas': [10**i for i in range(1,5)]},  
    
#     'model abs 3x3 bp 224 1000 filters':{
#                 'iden':'model_abs_3x3_bp_224',
#                 'model':EngModel3LAbsBP(filters_3 = 1000, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessGS, 
#                 'num_layers':3,
#                 'num_features':1000,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas': [10**i for i in range(1,5)]},  
    
#     'model abs 3x3 bp 224 10000 filters':{
#                 'iden':'model_abs_3x3_bp_224',
#                 'model':EngModel3LAbsBP(filters_3 = 10000, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessGS, 
#                 'num_layers':3,
#                 'num_features':10000,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas': [10**i for i in range(1,5)]},  

    
        # 'model abs 3x3 bp 224 10 filters':{
        #         'iden':'model_abs_3x3_bp_224',
        #         'model':EngModel3LAbsBP(filters_3 = 10, global_mp = MAX_POOL).Build(),
        #         'layers': ['last'], 
        #         'preprocess':Preprocess(im_size=224).PreprocessGS, 
        #         'num_layers':3,
        #         'num_features':10,
        #         'dim_reduction_type':None,
        #         'max_pool':MAX_POOL,
        #         'alphas': [10**i for i in range(2,5)]},  

    
    
#         'model abs 3x3 bp 224 100 filters':{
#                 'iden':'model_abs_3x3_bp_224',
#                 'model':EngModel3LAbsBP(filters_3 = 100, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessGS, 
#                 'num_layers':3,
#                 'num_features':100,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas': [10**i for i in range(2,5)]},  
        
    
         
#         'model abs 3x3 bp 224 1000 filters':{
#                 'iden':'model_abs_3x3_bp_224',
#                 'model':EngModel3LAbsBP(filters_3 = 1000, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessGS, 
#                 'num_layers':3,
#                 'num_features':1000,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas': [10**i for i in range(2,5)]},    


#         'model abs 3x3 bp 224 10000 filters':{
#                 'iden':'model_abs_3x3_bp_224',
#                 'model':EngModel3LAbsBP(filters_3 = 10000, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessGS, 
#                 'num_layers':3,
#                 'num_features':10000,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas': [10**i for i in range(2,5)]},  


#         'model abs 3x3 bp 224 100000 filters':{
#                 'iden':'model_abs_3x3_bp_224',
#                 'model':EngModel3LAbsBP(filters_3 = 10000, batches_3 = 10, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessGS, 
#                 'num_layers':3,
#                 'num_features':100000,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas': [10**i for i in range(2,5)]}, 
    
    
#        'alexnet u wide 10 filters':{
#                 'iden':'alexnet_u_wide',
#                 'model':AlexnetUWide(filters_5 = 10, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#                 'num_layers':5,
#                 'num_features':10,
#                 'dim_reduction_type':None,
#                  'max_pool':MAX_POOL,
#                 'alphas': [10**i for i in range(1,4)]},  
        
    
         
#        'alexnet u wide 100 filters':{
#                 'iden':'alexnet_u_wide',
#                 'model':AlexnetUWide(filters_5 = 100, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#                 'num_layers':5,
#                 'num_features':100,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas': [10**i for i in range(1,4)]},     
    
    
#        'alexnet u wide 1000 filters':{
#                 'iden':'alexnet_u_wide',
#                 'model':AlexnetUWide(filters_5 = 1000, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#                 'num_layers':5,
#                 'num_features':1000,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas': [10**i for i in range(1,4)]},  

    
    
#        'alexnet u wide 10000 filters':{
#                 'iden':'alexnet_u_wide',
#                 'model':AlexnetUWide(filters_5 = 10000, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#                 'num_layers':5,
#                 'num_features':10000,
#                 'dim_reduction_type':None,
#                  'max_pool':MAX_POOL,
#                 'alphas': [10**i for i in range(1,4)]},   
        
    
#        'alexnet u wide 100000 filters':{
#                 'iden':'alexnet_u_wide',
#                 'model':AlexnetUWide(filters_5 = 10000, batches_5 = 10, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#                 'num_layers':5,
#                 'num_features':100000,
#                 'dim_reduction_type':None,
#                  'max_pool':MAX_POOL,
#                 'alphas': [10**i for i in range(1,4)]}, 

    
       'alexnet conv1 ':{
                'iden':'alexnet_conv1',
                'model':Alexnet(features_layer =2, global_mp = MAX_POOL).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                'num_layers':1,
                'num_features':64,
                'dim_reduction_type':None,
                'max_pool':MAX_POOL,
                'alphas': [10**i for i in range(3,8)]},            
    
       'alexnet conv2 ':{
                'iden':'alexnet_conv2',
                'model':Alexnet(features_layer =5, global_mp = MAX_POOL).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                'num_layers':2,
                'num_features':192,
                'dim_reduction_type':None,
                'max_pool':MAX_POOL,
                'alphas': [10**i for i in range(3,8)]},      
    
       'alexnet conv3 ':{
                'iden':'alexnet_conv3',
                'model':Alexnet(features_layer =7, global_mp = MAX_POOL).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                'num_layers':3,
                'num_features':384,
                'dim_reduction_type':None,
                'max_pool':MAX_POOL,
                'alphas': [10**i for i in range(3,9)]},    
    
       'alexnet conv4 ':{
                'iden':'alexnet_conv4',
                'model':Alexnet(features_layer =9, global_mp = MAX_POOL).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                'num_layers':4,
                'num_features':256,
                'dim_reduction_type':None,
                'max_pool':MAX_POOL,
                'alphas': [10**i for i in range(3,7)]},    
    
       'alexnet u conv1':{
                'iden':'alexnet_u_conv1',
                'model':AlexnetU(features_layer =2, global_mp = MAX_POOL).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                'num_layers':1,
                'num_features':64,
                'dim_reduction_type':None,
                'max_pool':MAX_POOL,
                'alphas': [10**i for i in range(1,6)]},   
       
        'alexnet u conv2':{
                'iden':'alexnet_u_conv2',
                'model':AlexnetU(features_layer =5, global_mp = MAX_POOL).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                'num_layers':2,
                'num_features':192,
                'dim_reduction_type':None,
                'max_pool':MAX_POOL,
                'alphas': [10**i for i in range(1,4)]},   
       
        'alexnet u conv3':{
                'iden':'alexnet_u_conv3',
                'model':AlexnetU(features_layer =7, global_mp = MAX_POOL).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                'num_layers':3,
                'num_features':384,
                'dim_reduction_type':None,
                'max_pool':MAX_POOL,
                'alphas': [10**i for i in range(1,4)]},   
       
        'alexnet u conv4':{
                'iden':'alexnet_u_conv4',
                'model':AlexnetU(features_layer =9, global_mp = MAX_POOL).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                'num_layers':4,
                'num_features':256,
                'dim_reduction_type':None,
                'max_pool':MAX_POOL,
                'alphas': [10**i for i in range(1,4)]},   
}
 


for region in REGIONS:
    
    for model_name, model_info in MODEL_DICT.items():

        print(model_name)

        activations_identifier = get_activations_iden(model_info, DATASET, MODE)
        print(activations_identifier)

        activations = Activations(model=model_info['model'],
                                layer_names=model_info['layers'],
                                dataset=DATASET,
                                preprocess=model_info['preprocess'],
                                mode = MODE,
                            )           
        
        activations.get_array(ACTIVATIONS_PATH,activations_identifier)   
        
        for alpha in model_info['alphas']:
            
            print('alpha',alpha)
            regression_model = Ridge(alpha=alpha)            
            scores_identifier = get_scores_iden(model_info, activations_identifier, region, DATASET, MODE, alpha)
            
            scorer(model_name=model_info['iden'],
                   activations_identifier=activations_identifier,
                   scores_identifier=scores_identifier,
                   regression_model=regression_model,
                   dataset=DATASET,
                   mode=MODE,
                   regions=[region])
            # ,
            #        n_dims = None,
            #        dim_reduction_type = None

#                   )
