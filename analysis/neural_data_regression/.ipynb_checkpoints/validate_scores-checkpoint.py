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


# define paths

BEST_ALPHA_PATH = '/data/atlas/regression_alphas'    
ACTIVATIONS_PATH = '/data/atlas/activations'

# define constants
DATASET = 'naturalscenes'
REGIONS = ['V1','V2','V3','V4']
# DATASET = 'majajhong'
# REGIONS = ['V4','IT']

MAX_POOL = True
MODE = 'test'







MODEL_DICT= {
    
       'alexnet conv1':{
                'iden':'alexnet_conv1',
                #'model':Alexnet(features_layer =2, global_mp = MAX_POOL).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                'num_layers':1,
                'num_features':64,
                'dim_reduction_type':None,
                'max_pool':MAX_POOL,
                'alphas': 'alexnet_mp'},            
    
       'alexnet conv2':{
                'iden':'alexnet_conv2',
                #'model':Alexnet(features_layer =5, global_mp = MAX_POOL).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                'num_layers':2,
                'num_features':192,
                'dim_reduction_type':None,
                'max_pool':MAX_POOL,
                'alphas': 'alexnet_mp'},      
    
       'alexnet conv3':{
                'iden':'alexnet_conv3',
                #'model':Alexnet(features_layer =7, global_mp = MAX_POOL).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                'num_layers':3,
                'num_features':384,
                'dim_reduction_type':None,
                'max_pool':MAX_POOL,
                'alphas': 'alexnet_mp'},    
    
       'alexnet conv4':{
                'iden':'alexnet_conv4',
               #'model':Alexnet(features_layer =9, global_mp = MAX_POOL).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                'num_layers':4,
                'num_features':256,
                'dim_reduction_type':None,
                'max_pool':MAX_POOL,
                'alphas': 'alexnet_mp'},     
        
       'alexnet':{
                'iden':'alexnet',
               #'model':Alexnet(global_mp = MAX_POOL).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                'num_layers':5,
                'num_features':256,
                'dim_reduction_type':None,
                'max_pool':MAX_POOL,
                'alphas': 'alexnet_mp'},    
    
       'alexnet u conv1':{
                'iden':'alexnet_u_conv1',
                #'model':AlexnetU(features_layer =2, global_mp = MAX_POOL).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                'num_layers':1,
                'num_features':64,
                'dim_reduction_type':None,
                'max_pool':MAX_POOL,
                'alphas': 'alexnet_u_mp'},   
       
        'alexnet u conv2':{
                'iden':'alexnet_u_conv2',
                #'model':AlexnetU(features_layer =5, global_mp = MAX_POOL).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                'num_layers':2,
                'num_features':192,
                'dim_reduction_type':None,
                'max_pool':MAX_POOL,
                'alphas': 'alexnet_u_mp'},   
       
        'alexnet u conv3':{
                'iden':'alexnet_u_conv3',
                #'model':AlexnetU(features_layer =7, global_mp = MAX_POOL).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                'num_layers':3,
                'num_features':384,
                'dim_reduction_type':None,
                'max_pool':MAX_POOL,
                'alphas': 'alexnet_u_mp'},   
       
        'alexnet u conv4':{
                'iden':'alexnet_u_conv4',
                #'model':AlexnetU(features_layer =9, global_mp = MAX_POOL).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                'num_layers':4,
                'num_features':256,
                'dim_reduction_type':None,
                'max_pool':MAX_POOL,
                'alphas': 'alexnet_u_mp'},       
        
        'alexnet u':{
                'iden':'alexnet_u',
                #'model':AlexnetU(global_mp = MAX_POOL).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                'num_layers':5,
                'num_features':256,
                'dim_reduction_type':None,
                'max_pool':MAX_POOL,
                'alphas': 'alexnet_u_mp'},     
    
    # 'scattering transform kymatio':{
    #             'iden':'scat_transform_kymatio_J3_L4',
    #             #'model':ScatTransformKymatio(J = 3, L = 8, M = 32, N = 32, flatten = True, global_mp= False).Build(),
    #             'layers': ['last'], 
    #             'preprocess':Preprocess(im_size=32).PreprocessGS, 
    #             'num_layers':2,
    #             'num_features':'x',
    #             'dim_reduction_type':None,
    #             'max_pool':False,
    #             'alphas': 'scat_transform_kymatio_J3_L4'},
    
    
#     'scattering transform kymatio':{
#                 'iden':'scat_transform_kymatio_J3_L4_rgb',
#                 #'model':ScatTransformKymatio(J = 3, L = 8, M = 32, N = 32, flatten = True, global_mp= False).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=32).PreprocessRGB, 
#                 'num_layers':2,
#                 'num_features':'x',
#                 'dim_reduction_type':None,
#                 'max_pool':False,
#                 'alphas': 'scat_transform_kymatio_J3_L4_rgb'},
    
    
#         'model abs 3x3 bp 224 10 filters':{
#                 'iden':'model_abs_3x3_bp_224',
#                 #'model':EngModel3LAbsBP(filters_3 = 10, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessGS, 
#                 'num_layers':3,
#                 'num_features':10,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas': 'model_abs_3x3_bp_224_mp'},  
    
#     'model abs 3x3 bp 224 100 filters':{
#                 'iden':'model_abs_3x3_bp_224',
#                 #'model':EngModel3LAbsBP(filters_3 = 100, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessGS, 
#                 'num_layers':3,
#                 'num_features':100,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas': 'model_abs_3x3_bp_224_mp'},  
    
#     'model abs 3x3 bp 224 1000 filters':{
#                 'iden':'model_abs_3x3_bp_224',
#                 #'model':EngModel3LAbsBP(filters_3 = 1000, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessGS, 
#                 'num_layers':3,
#                 'num_features':1000,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas': 'model_abs_3x3_bp_224_mp'},  
    
#     'model abs 3x3 bp 224 10000 filters':{
#                 'iden':'model_abs_3x3_bp_224',
#                 #'model':EngModel3LAbsBP(filters_3 = 10000, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessGS, 
#                 'num_layers':3,
#                 'num_features':10000,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas': 'model_abs_3x3_bp_224_mp'},  
    
#     'model abs 3x3 bp 224 ap 10 filters':{
#                 'iden':'model_abs_3x3_bp_224_ap',
#                 #'model':EngModel3LAbsBPAP(filters_3 = 10, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessGS, 
#                 'num_layers':3,
#                 'num_features':10,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas': 'model_abs_3x3_bp_224_ap_mp'},  
    
#     'model abs 3x3 bp 224 ap 100 filters':{
#                 'iden':'model_abs_3x3_bp_224_ap',
#                 #'model':EngModel3LAbsBPAP(filters_3 = 100, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessGS, 
#                 'num_layers':3,
#                 'num_features':100,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas': 'model_abs_3x3_bp_224_ap_mp'},  
    
#     'model abs 3x3 bp 224 ap 1000 filters':{
#                 'iden':'model_abs_3x3_bp_224_ap',
#                 #'model':EngModel3LAbsBPAP(filters_3 = 1000, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessGS, 
#                 'num_layers':3,
#                 'num_features':1000,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas': 'model_abs_3x3_bp_224_ap_mp'},  
    
#     'model abs 3x3 bp 224 ap 10000 filters':{
#                 'iden':'model_abs_3x3_bp_224_ap',
#                 #'model':EngModel3LAbsBPAP(filters_3 = 10000, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessGS, 
#                 'num_layers':3,
#                 'num_features':10000,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas': 'model_abs_3x3_bp_224_ap_mp'},  

   
#        'alexnet u wide 10 filters':{
#                 'iden':'alexnet_u_wide',
#                 #'model':AlexnetUWide(filters_5 = 10, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#                 'num_layers':5,
#                 'num_features':10,
#                 'dim_reduction_type':None,
#                  'max_pool':MAX_POOL,
#                 'alphas': 'alexnet_u_wide_mp'},  
        
    
         
#        'alexnet u wide 100 filters':{
#                 'iden':'alexnet_u_wide',
#                 #'model':AlexnetUWide(filters_5 = 100, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#                 'num_layers':5,
#                 'num_features':100,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas': 'alexnet_u_wide_mp'},     
    
    
#        'alexnet u wide 1000 filters':{
#                 'iden':'alexnet_u_wide',
#                 #'model':AlexnetUWide(filters_5 = 1000, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#                 'num_layers':5,
#                 'num_features':1000,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas': 'alexnet_u_wide_mp'},  

    
    
#        'alexnet u wide 10000 filters':{
#                 'iden':'alexnet_u_wide',
#                 #'model':AlexnetUWide(filters_5 = 10000, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#                 'num_layers':5,
#                 'num_features':10000,
#                 'dim_reduction_type':None,
#                  'max_pool':MAX_POOL,
#                 'alphas': 'alexnet_u_wide_mp'},   
        
    
#        'alexnet':{
#                 'iden':'alexnet',
#                 #'model':Alexnet(global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#                 'num_layers':5,
#                 'num_features':256,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas': 'alexnet_mp'},           
    
    
#        'alexnet u':{
#                 'iden':'alexnet_u',
#                 #'model':AlexnetU(global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#                 'num_layers':5,
#                 'num_features':256,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas': 'alexnet_u_mp'},  

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



 

