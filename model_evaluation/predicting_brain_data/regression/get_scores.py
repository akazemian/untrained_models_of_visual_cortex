# getting activations for a specific dataset from a specific model. Output is an xarray with dims: features x presentation (stimulus_id)
# from kymatio.torch import Scattering2D
import os 
import sys
ROOT_DIR = os.getenv('MB_ROOT_PATH')
sys.path.append(ROOT_DIR)

import xarray as xr
import torch
import torchvision
import warnings
warnings.filterwarnings('ignore')
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
import random
from models.all_models.model_3L_abs_blurpool_avgpool import ExpansionModel
from models.all_models.model_3L_abs_avgpool import ModelAbsAP
from models.all_models.alexnet import Alexnet
from models.all_models.alexnet_u import AlexnetU

from models.all_models.model_3L_linear import LinearModel
from models.all_models.fully_connected import FCModel
from models.all_models.fully_connected_3layers import FCModel3L
from models.all_models.fully_random import FRModel
from tools.processing import *
from tools.utils import get_activations_iden, get_scores_iden
from analysis.encoding_model_analysis.tools.extractor import Activations
from analysis.encoding_model_analysis.tools.regression import *
from analysis.encoding_model_analysis.tools.scorer import *


torch.manual_seed(0)
torch.cuda.manual_seed(0)
untrained_alexnet = torchvision.models.alexnet(pretrained=False)
alexnet_pytorch =  torchvision.models.alexnet(pretrained=True)

ROOT = os.getenv('MB_DATA_PATH')
ACTIVATIONS_PATH = os.path.join(ROOT,'activations')   

DATASET = 'naturalscenes'
REGIONS = ['general']

# DATASET = 'majajhong'
# REGIONS = ['V4','IT']

MAX_POOL = True
MODE = 'ridgecv'
HOOK = None



alphas = [10**i for i in range(1,10)]

MODEL_DICT = {               
    
    

#     'fully_random':{
#                 'iden':'expansion_model_final_random',
#                 'model':FRModel(filters_1=36).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#                 'num_layers':3,
#                 'num_features':10000,
#                 'dim_reduction_type':None,
#                 'n_dims': None,
#                 'max_pool':MAX_POOL,
#                 'alphas':alphas
#                 }, 
    
#     'gabor model':{
#                 'iden':'expansion_model_final_gabor_corrected_3_scales',
#                 'model':ExpansionModel(filter_params={'type':'gabor','n_ories':12,'num_scales':3}).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#                 'n_dims': None,
#                 'num_layers':3,
#                 'num_features':10000,
#                 'max_pool':MAX_POOL,
#                 'alphas':alphas
#                 },  
    
 
    
    

   
    
#################################################################################
#     'expansion model 3L 10':{
#                 'iden':'expansion_model_final',
#                 'model':ExpansionModel(filters_3=10).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#                 'num_layers':3,
#                 'num_features':10,
#                 'max_pool':MAX_POOL,
#                 'alphas':alphas},   

    
#     'expansion model 3L 100':{
#                 'iden':'expansion_model_final',
#                 'model':ExpansionModel(filters_3=100).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#                 'num_layers':3,
#                 'num_features':100,
#                 'max_pool':MAX_POOL,
#                 'alphas':alphas}, 
    
    
    
#     'expansion model 3L 1000':{
#                 'iden':'expansion_model_final',
#                 'model':ExpansionModel(filters_3=1000, gpool=MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#                 'num_layers':3,
#                 'num_features':1000,
#                 'max_pool':MAX_POOL,
#                 'alphas':alphas}, 
    
    
    'expansion model 3L 10000':{
                'iden':'expansion_model_final',
                'model':ExpansionModel(filters_3=10000).Build(),
                'layers': ['last'], 
                'preprocess':Preprocess(im_size=224).PreprocessRGB, 
                'num_layers':3,
                'num_features':10000,
                'max_pool':MAX_POOL,
                'alphas':alphas}, 
        
    
    # 'expansion model 3L 100000':{
    #             'iden':'expansion_model_final',
    #             'model':ExpansionModel(batches_3=10,filters_3=10000).Build(),
    #             'layers': ['last'], 
    #             'preprocess':Preprocess(im_size=224).PreprocessRGB, 
    #             'num_layers':3,
    #             'num_features':100000,
    #             'max_pool':MAX_POOL,
    #             'alphas':alphas}, 
    
    
    
    
 #################################   
    
        
#        'alexnet conv1':{
#                 'iden':'alexnet_conv1',
#                 'model':Alexnet(features_layer =2).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#                 'num_layers':1,
#                 'num_features':64,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas':alphas},            
    
#        'alexnet conv2':{
#                 'iden':'alexnet_conv2',
#                 'model':Alexnet(features_layer =5).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#                 'num_layers':2,
#                 'num_features':192,44676
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas':alphas},      
    
#        'alexnet conv3':{
#                 'iden':'alexnet_conv3',
#                 'model':Alexnet(features_layer =7).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#                 'num_layers':3,
#                 'num_features':384,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas':alphas},    
    
#        'alexnet conv4':{
#                 'iden':'alexnet_conv4',
#                'model':Alexnet(features_layer =9).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#                 'num_layers':4,
#                 'num_features':256,
#                 'dim_reduction_type':None,44676
#                 'max_pool':MAX_POOL,
#                 'alphas':alphas},     
        
       # 'alexnet conv5':{
       #          'iden':'alexnet_conv5',
       #          'model':Alexnet().Build(),
       #          'layers': ['last'], 
       #          'preprocess':Preprocess(im_size=224).PreprocessRGB, 
       #          'num_layers':5,
       #          'num_features':256,
       #          'dim_reduction_type':None,
       #          'max_pool':MAX_POOL,
       #          'alphas':alphas},    
    
    
    
##############################################
    
    
#        'alexnet u conv1':{
#                 'iden':'alexnet_u_conv1',
#                 'model':AlexnetU(features_layer =2).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#                 'num_layers':1,44676
#                 'num_features':64,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas':alphas},   
       
#         'alexnet u conv2':{
#                 'iden':'alexnet_u_conv2',
#                 'model':AlexnetU(features_layer =5).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#                 'num_layers':2,
#                 'num_features':192,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas':alphas},   
       
#         'alexnet u conv3':{
#                 'iden':'alexnet_u_conv3',
#                 'model':AlexnetU(features_layer =7).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#                 'num_layers':3,
#                 'num_features':384,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas':alphas},   
       
#         'alexnet u conv4':{
#                 'iden':'alexnet_u_conv4',
#                 'model':AlexnetU(features_layer =9).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=224).PreprocessRGB, 
#                 'num_layers':4,
#                 'num_features':256,
#                 'dim_reduction_type':None,
#                 'max_pool':MAX_POOL,
#                 'alphas':alphas},       
        
        # 'alexnet u conv5':{
        #         'iden':'alexnet_u_conv5',
        #         'model':AlexnetU().Build(),
        #         'layers': ['last'], 
        #         'preprocess':Preprocess(im_size=224).PreprocessRGB, 
        #         'num_layers':5,
        #         'num_features':256,
        #         'dim_reduction_type':None,
        #         'max_pool':MAX_POOL,
        #         'alphas':alphas},   44676
    
    
###################################################################    
    
    
    
    
    #        'scattering kymatio j2':{
#                 'iden':'st_kymatio_j2',
#                 'model':ScatTransformKymatio(J=2, L=8, M=32, N=32, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=32).PreprocessRGB, 
#                 'num_layers':2,
#                 'num_features':'n',
#                 'max_pool':MAX_POOL,
#                 'alphas':[10**i for i in range(10)]},   

    
#        'scattering kymatio j3':{
#                 'iden':'st_kymatio_j3',
#                 'model':ScatTransformKymatio(J=3, L=8, M=32, N=32, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=32).PreprocessRGB, 
#                 'num_layers':2,
#                 'num_features':'n',
#                 'max_pool':MAX_POOL,
#                 'alphas':[10**i for i in range(10)]},   

#        'scattering kymatio j4':{
#                 'iden':'st_kymatio_j4',
#                 'model':ScatTransformKymatio(J=4, L=8, M=32, N=32, global_mp = MAX_POOL).Build(),
#                 'layers': ['last'], 
#                 'preprocess':Preprocess(im_size=32).PreprocessRGB, 
#                 'num_layers':2,
#                 'num_features':'n',
#                 'max_pool':MAX_POOL,
#                 'alphas':[10**i for i in range(10)]},        
}
 


for region in REGIONS:
    
    print(region)
    
    for model_name, model_info in MODEL_DICT.items():

        print(model_name)

        activations_identifier = get_activations_iden(model_info, DATASET, MODE)
        print(activations_identifier)

        activations = Activations(model=model_info['model'],
                                layer_names=model_info['layers'],
                                dataset=DATASET,
                                preprocess=model_info['preprocess'],
                                mode = MODE,
                                _hook = HOOK,
                                batch_size = 100
                            )           
        
        activations.get_array(ACTIVATIONS_PATH,activations_identifier)   
    
        scores_identifier = get_scores_iden(model_info, activations_identifier, region, DATASET, MODE)
        scores_identifier = scores_identifier + '_normalized_features'
        scorer(model_name=model_info['iden'],
               activations_identifier=activations_identifier,
               scores_identifier=scores_identifier,
               dataset=DATASET,
               mode=MODE,
               region=region,
            alpha_values=model_info['alphas'])

